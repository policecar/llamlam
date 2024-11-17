"""
Based on https://github.com/cloneofsimo/min-max-gpt/blob/main/train.py
without distributed training (zero optimization, etc.)

Usage:
export PYTORCH_ENABLE_MPS_FALLBACK=1
deepspeed --num_gpus $(nvidia-smi -L | wc -l) llamlam/train.py --batch_size 16 --learning_rate 1e-5 --run_name "test"

ran: deepspeed --num_gpus $(nvidia-smi -L | wc -l) llamlam/train.py --learning_rate 1e-4 --head_width 32 --run_name "test"
     with default batch_size 16

"""

import json
import math
import os

import click
import deepspeed
import torch
import wandb

from pathlib import Path

from deepspeed import get_accelerator
from deepspeed.utils import logger
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, default_data_collator, get_scheduler

from llamlam.data import CustomDataset
from llamlam.model import Config, GPTModel
from llamlam.utils import set_seed


def train(model_engine, train_loader, device):
    """ """
    model_engine.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        # optimizer.zero_grad()  # reset loss gradients, handled by model_engine
        # send data to device and convert to desired data type
        input_ids = batch["input_ids"].to(device)
        # forward pass
        outputs = model_engine(input_ids)
        loss = outputs["loss"]
        total_loss += loss.item()

        logger.info(f"loss : {loss.item()}")
        wandb.log({"trainloss": loss.item()})

        model_engine.backward(loss)  # run backpropagation
        model_engine.step()  # update parameters and learning rate, zeros gradients
        # get_accelerator().empty_cache()  # clear cache

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model_engine, val_loader, device):
    """ """
    model_engine.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            outputs = model_engine(input_ids)
            loss = outputs["loss"]
            total_loss += loss.float()

    losses = total_loss / len(val_loader)

    try:
        perplexity = torch.exp(losses).item()  # type: ignore
    except OverflowError:
        perplexity = float("inf")

    return losses, perplexity


def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    output_model_file = os.path.join(save_dir, "model.pt")

    # model_to_save = model.module if hasattr(model, "module") else model
    # torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(model.state_dict(), output_model_file)


def load_model(model, save_dir):
    output_model_file = os.path.join(save_dir, "model.pt")
    model.load_state_dict(torch.load(output_model_file))


@click.command()
@click.option("--num_warmup_steps", default=0, help="Number of warmup steps")
@click.option("--seed", default=42, help="Random seed")
@click.option("--output_dir", default="experiments", help="Output directory")
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
    local_rank,  # needed for deepspeed
    debug,
    run_name,
    n_layer,
    n_head,
    head_width,
    batch_size,
    learning_rate,
    weight_decay,
    num_train_epochs,
    lr_scheduler_type,
):
    set_seed(seed)  # set seed first thing

    if run_name is None:
        run_name = f"LR:{learning_rate}_HeadWidth:{head_width}_TotalBS:{batch_size}_Nhead:{n_head}_NLayer:{n_layer}"

    output_dir = Path(__file__).resolve().parent.parent / "data" / "output" / run_name
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(get_accelerator().device_name())  # type: ignore

    train_config = {
        "train_batch_size": batch_size,
        "bfloat16": {"enabled": False},  # True
        "gradient_clipping": 1.0,
    }

    # Initialize WandB
    wandb.init(
        name=run_name,
        config={
            "seed": seed,
            "learning_rate": learning_rate,
            "lr_scheduler_type": lr_scheduler_type,
            "num_warmup_steps": num_warmup_steps,
            "weight_decay": weight_decay,
            "num_train_epochs": num_train_epochs,
            "head_width": head_width,
            "Nhead": n_head,
            "NLayer": n_layer,
            "output_dir": output_dir,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = Config(
        vocab_size=len(tokenizer),
        max_length=512,
        n_head=n_head,
        n_layer=n_layer,
        n_embd=n_head * head_width,
    )

    model = GPTModel(config)
    # model.train()

    # Declare datasets, samplers, and dataloaders
    train_dataset = CustomDataset(tokenizer, "train", debug=debug)
    val_dataset = CustomDataset(tokenizer, "validation", debug=debug)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        sampler=val_sampler,
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

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95))

    # Define learning rate scheduler
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_epochs * math.ceil(len(train_loader)),
    )

    # Initialize distributed backend
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, config=train_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )

    # Training loop
    for epoch in range(num_train_epochs):
        avg_train_loss = train(model_engine, train_loader, model_engine.device)
        val_loss, perp = validate(model_engine, val_loader, device=device)
        logger.info(
            f"Epoch {epoch+1}, train loss, validation loss: {avg_train_loss}, {val_loss}"
        )
        wandb.log({"ppl": perp, "loss": val_loss, "epoch": epoch})

        model_output_dir = os.path.join(output_dir, f"step_{epoch}_final")
        save_model(model_engine, model_output_dir)


if __name__ == "__main__":
    main()
