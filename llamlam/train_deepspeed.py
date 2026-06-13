"""
DeepSpeed training entry point (single- or multi-GPU), mirroring train.py.

Originally based on https://github.com/cloneofsimo/min-max-gpt/blob/main/train.py

Usage:
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  deepspeed --num_gpus $(nvidia-smi -L | wc -l) -m llamlam.train_deepspeed \
      --batch_size 16 --learning_rate 1e-4 --run_name test

  # GPTModel instead of the DiffTransformer:
  deepspeed --num_gpus 1 -m llamlam.train_deepspeed --model_type gpt --run_name test-gpt
"""

import os

from pathlib import Path

import click
import deepspeed
import torch
import wandb

from deepspeed import get_accelerator
from deepspeed.utils import logger
from datasets import load_dataset
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

from llamlam.config import Config
from llamlam.data import DataCollator
from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel
from llamlam.utils import get_grouped_params, set_seed


def train(model_engine, train_loader, device):
    model_engine.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        total_loss += loss.item()

        logger.info(f"loss : {loss.item()}")
        wandb.log({"trainloss": loss.item()})

        model_engine.backward(loss)  # run backpropagation
        model_engine.step()  # update parameters and lr, then zero gradients

    return total_loss / len(train_loader)


def validate(model_engine, val_loader, device):
    model_engine.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model_engine(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            total_loss += outputs["loss"].float()

    losses = total_loss / len(val_loader)
    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")

    return losses, perplexity


def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))


@click.command()
@click.option("--seed", default=42, help="Random seed")
@click.option("--run_name", default=None, help="Run name")
@click.option("--local_rank", default=-1, help="Local rank (set by deepspeed)")
@click.option(
    "--model_type",
    default="diff",
    type=click.Choice(["diff", "gpt"]),
    help="Model architecture",
)
@click.option("--n_layers", default=12, help="Number of layers")
@click.option("--n_heads", default=12, help="Number of heads")
@click.option("--dim_head", default=64, help="Dimension of each attention head")
@click.option("--batch_size", default=16, help="Per-GPU micro batch size")
@click.option("--learning_rate", default=1e-3, help="Learning rate")
@click.option("--weight_decay", default=0.1, help="Weight decay")
@click.option("--n_epochs", default=1, help="Number of training epochs")
@click.option("--lr_scheduler_type", default="linear", help="LR scheduler type")
@click.option("--n_warmup_steps", default=0, help="Number of warmup steps")
@click.option("--bf16", is_flag=True, help="Enable bfloat16 training")
def main(
    seed,
    run_name,
    local_rank,  # needed for deepspeed
    model_type,
    n_layers,
    n_heads,
    dim_head,
    batch_size,
    learning_rate,
    weight_decay,
    n_epochs,
    lr_scheduler_type,
    n_warmup_steps,
):
    set_seed(seed)

    if run_name is None:
        run_name = f"ds_{model_type}_LR{learning_rate}_BS{batch_size}_L{n_layers}_H{n_heads}"

    output_dir = Path(__file__).resolve().parent.parent / "data" / "output" / run_name
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(get_accelerator().device_name())

    # DeepSpeed engine config. train_micro_batch_size_per_gpu is the DataLoader
    # batch size; the global batch is inferred across GPUs / accumulation steps.
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": bf16},
        "gradient_clipping": 1.0,
    }

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Config(
        vocab_size=len(tokenizer),
        model_type=model_type,
        n_layers=n_layers,
        n_heads=n_heads,
        dim_head=dim_head,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
    )

    wandb.init(
        project="llamlam",
        name=run_name,
        config={**config.__dict__},
    )

    model = GPTModel(config) if model_type == "gpt" else DiffTransformer(config)

    # Data: same pipeline as train.py.
    dataset = load_dataset(
        path=config.data_path,
        name=config.data_name if config.data_files is None else None,
        data_files=config.data_files,
    )
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=min(2560, int(0.1 * len(dataset["train"])))
        )
    val_split = "validation" if "validation" in dataset else "test"
    dataset = dataset.select_columns(["text"])

    collate_fn = DataCollator(tokenizer, config)
    train_loader = DataLoader(
        dataset["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    val_loader = DataLoader(
        dataset[val_split],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size * 2,
    )

    optimizer = AdamW(
        get_grouped_params(model, weight_decay=weight_decay, no_decay=config.no_decay),
        lr=learning_rate,
        betas=(0.9, 0.95),
    )

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=n_warmup_steps,
        num_training_steps=n_epochs * len(train_loader),
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model, config=ds_config, lr_scheduler=lr_scheduler, optimizer=optimizer
    )

    for epoch in range(n_epochs):
        avg_train_loss = train(model_engine, train_loader, model_engine.device)
        val_loss, perplexity = validate(model_engine, val_loader, device=device)
        logger.info(
            f"Epoch {epoch + 1}, train loss, validation loss: {avg_train_loss}, {val_loss}"
        )
        wandb.log({"ppl": perplexity, "val_loss": val_loss, "epoch": epoch})

        save_model(model_engine, os.path.join(output_dir, f"step_{epoch}_final"))


if __name__ == "__main__":
    main()
