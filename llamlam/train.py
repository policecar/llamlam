import json
import math
import numpy as np
import os
import random

import click
import deepspeed
import torch
import wandb

from datasets import load_dataset
from deepspeed import get_accelerator
from deepspeed.utils import logger
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, default_data_collator, get_scheduler

from model import LMConfig, GPTModel

# very much based on https://github.com/cloneofsimo/min-max-gpt/blob/main/train.py
# without the distributed bit


class CustomDataset(Dataset):
    def __init__(self, tokenizer, type_path="train", max_length=512, debug=False):
        if debug:
            vernum = 2
        else:
            vernum = 103
        self.vernum = vernum
        self.dataset = load_dataset(
            "wikitext", f"wikitext-{vernum}-raw-v1", split=type_path
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return int(len(self.dataset) * 0.1) if (self.vernum == 103) else 32

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        # logger.info(text)
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.squeeze()}


def train(ds_engine, train_loader, device):
    ds_engine.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        outputs = ds_engine(input_ids)
        loss = outputs["loss"]
        total_loss += loss.item()

        logger.info(f"loss : {loss.item()}")
        wandb.log({"trainloss": loss.item()})

        ds_engine.backward(loss)
        ds_engine.step()
        get_accelerator().empty_cache()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            loss = outputs["loss"]
            total_loss += loss.float()

    losses = total_loss / len(val_loader)

    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")

    model.train()

    return losses, perplexity


def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    output_model_file = os.path.join(save_dir, "model.pt")

    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), output_model_file)


@click.command()
@click.option("--num_warmup_steps", default=0, help="Number of warmup steps")
@click.option("--seed", default=42, help="Random seed")
@click.option("--output_dir", default="experiments", help="Output directory")
@click.option("--offload", default=True, help="Offload computation")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("--run_name", default=None, help="Run name")
@click.option("--local_rank", default=-1, help="Local rank")
@click.option("--n_layer", default=12, help="Number of layers")
@click.option("--n_head", default=4, help="Number of heads")
@click.option(
    "--head_width",
    default=2,
    help="Width of the head, total dim is head_width * n_head",
)
@click.option("--batch_size", default=2048, help="Total training batch size")
@click.option("--learning_rate", default=1e-3, help="Learning rate")
@click.option("--weight_decay", default=0.1, help="Weight decay for optimization")
@click.option("--num_train_epochs", default=1, help="Number of training epochs")
@click.option(
    "--lr_scheduler_type", default="linear", help="Type of learning rate scheduler"
)
def main(
    num_warmup_steps,
    seed,
    output_dir,
    offload,
    debug,
    run_name,
    local_rank,
    n_layer,
    n_head,
    head_width,
    batch_size,
    learning_rate,
    weight_decay,
    num_train_epochs,
    lr_scheduler_type,
):
    # set seed first thing
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    if run_name is None:
        run_name = f"LR:{learning_rate}_HeadWidth:{head_width}_TotalBS:{batch_size}_Nhead:{n_head}_NLayer:{n_layer}"

    output_dir = Path().resolve(__file__) / "data" / output_dir / run_name
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(get_accelerator().device_name())

    ds_config = {
        "train_batch_size": batch_size,
        "bfloat16": {"enabled": False},  # True
        "gradient_clipping": 1.0,
    }

    # Initialize wandb
    wandb.init(
        name=run_name,
        config={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_train_epochs": num_train_epochs,
            "lr_scheduler_type": lr_scheduler_type,
            "num_warmup_steps": num_warmup_steps,
            "seed": seed,
            "output_dir": output_dir,
            "head_width": head_width,
            "Nhead": n_head,
            "NLayer": n_layer,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = LMConfig(
        vocab_size=len(tokenizer),
        max_length=512,
        n_head=n_head,
        n_layer=n_layer,
        n_embd=n_head * head_width,
    )

    # # zero-init
    # with deepspeed.zero.Init():
    model = GPTModel(config)

    model.train()

    train_dataset = CustomDataset(tokenizer, "train", debug=debug)
    val_dataset = CustomDataset(tokenizer, "validation", debug=debug)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        sampler=eval_sampler,
        batch_size=batch_size * 2,
    )

    # Config for no-decay
    no_decay_name_list = [
        "bias",
        "ln_",
        "ln_f.weight",
    ]

    optimizer_grouped_parameters = []
    final_optimizer_settings = {}

    for n, p in model.named_parameters():
        group_parameters = {}
        if p.requires_grad:
            if any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["weight_decay"] = 0.0
            else:
                group_parameters["weight_decay"] = weight_decay

            # Define learning rate for specific types of params

            is_embed = "embed" in n
            if "embed" in n or any(ndnl in n for ndnl in no_decay_name_list):
                group_parameters["lr"] = learning_rate * (3.3 if is_embed else 1.0)
            else:
                group_parameters["lr"] = learning_rate * (1 / head_width)

            group_parameters["params"] = [p]
            final_optimizer_settings[n] = {
                "lr": group_parameters["lr"],
                "wd": group_parameters["weight_decay"],
            }
            optimizer_grouped_parameters.append(group_parameters)

    # View the settings, see if anything is wrong.
    with open(os.path.join(output_dir, "opt_config.json"), "w") as json_file:
        json.dump(final_optimizer_settings, json_file, indent=4)

    AdamOptimizer = optim.Adam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95)
    )

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_epochs * math.ceil(len(train_loader)),
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, config=ds_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )

    for epoch in range(num_train_epochs):
        avg_train_loss = train(model_engine, train_loader, model_engine.device)
        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")
        eval_loss, perp = validate(model_engine, val_loader, device=device)
        logger.info(f"Eval loss : {eval_loss}")
        wandb.log({"ppl": perp, "loss": eval_loss, "epoch": epoch})

        model_output_dir = os.path.join(output_dir, f"step_{epoch}_final")
        save_model(model_engine, model_output_dir)


if __name__ == "__main__":
    main()
