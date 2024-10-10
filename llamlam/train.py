import argparse
import logging
import os
import yaml

from datetime import datetime
from pathlib import Path

import torch
import wandb

from accelerate import Accelerator
from datasets import load_dataset
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

from llamlam.config import Config
from llamlam.data import DataCollator
from llamlam.model import GPTModel
from llamlam.utils import evaluate, get_grouped_params, set_seed


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


if __name__ == "__main__":

    ##########################################
    # Setup
    ##########################################

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script for LlamLam")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory")
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument("--n_layer", type=int, help="Number of layers")
    parser.add_argument("--n_head", type=int, help="Number of heads")
    parser.add_argument(
        "--head_width",
        type=int,
        help="Width of the head, total dim is head_width * n_head",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Number of steps between evaluations",
    )
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        help="Type of learning rate scheduler",
    )
    parser.add_argument("--num_warmup_steps", type=int, help="Number of warmup steps")
    args = parser.parse_args()

    # Load default config
    config = Config()

    # Update config with parsed arguments
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            if hasattr(config, arg_name):
                setattr(config, arg_name, arg_value)
    logger.info(f"Arguments parsed: {vars(args)}")

    # Create run directory
    run_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(__file__).resolve().parent.parent / "data" / "runs" / run_name
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Run directory created at: {output_dir}")

    # Set seed
    set_seed(seed=config.seed)
    logger.info(f"Using seed: {config.seed}")

    accelerator = Accelerator()
    device = accelerator.device

    # device = torch.device(
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps" if torch.backends.mps.is_available() else "cpu"
    # )

    ##########################################
    # Initialize tokenizer & some
    ##########################################

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = config.max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = model.config.eos_token_id
    logger.info(f"Tokenizer initialized with vocabulary size: {len(tokenizer)}")

    # Update model config with tokenizer vocabulary size
    config.vocab_size = len(tokenizer)

    ##########################################
    # Init WandB, save config
    ##########################################

    # Init wandb
    wandb.init(
        project="llamlam",
        name=run_name,
        config={**config.__dict__},
    )

    # Save config to YAML file
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.__dict__, f)
    logger.info(f"Config saved to: {config_path}")

    ##########################################
    # Load data, make dataloaders
    ##########################################

    try:
        dataset = load_dataset(
            path=config.data_path,
            name=config.data_name if config.data_files is None else None,
            data_files=config.data_files,
        )
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")

    # If there's no validation set, make one by splitting off 10% of the training set
    if "validation" not in dataset:
        dataset = dataset["train"].train_test_split(
            test_size=min(2560, 0.1 * len(dataset["train"]))
        )

    # Drop all columns except 'text'
    dataset = dataset.select_columns(["text"])

    # Define collate function
    collate_fn = DataCollator(tokenizer, config)

    train_loader = DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=collate_fn,  # default_data_collator,
        batch_size=config.batch_size,
        num_workers=config.num_data_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset["test"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_data_workers,
        pin_memory=True,
    )

    ##########################################
    # Initialize model
    ##########################################

    model = GPTModel(config)
    # model.to(device)

    ##########################################
    # Define optimizer
    ##########################################

    optimizer = AdamW(
        get_grouped_params(model, weight_decay=config.weight_decay, no_decay=config.no_decay),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_epochs
        * len(train_loader)
        // config.gradient_accumulation_steps,
    )
    logger.info(f"Optimizer initialized with config: {config}")

    ##########################################
    # Training loop
    ##########################################

    # ~"no specific order, we just need to unpack objects in the same order we gave them to the prepare method"
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    val_losses = []
    tokens_seen = 0
    global_step = 0

    # TODO: add resumption of training from a checkpoint
    # https://github.com/huggingface/accelerate/blob/main/examples/complete_nlp_example.py#L175

    for epoch in range(config.num_epochs):
        train_loss = 0.0
        optimizer.zero_grad()  # reset gradients
        for step, batch in enumerate(train_loader):
            model.train()
            input_ids = batch["input_ids"]  # .to(device)
            # attention_mask = batch["attention_mask"]  # .to(device)
            loss = model(input_ids)["loss"]  # logits, (loss), (hidden_states)
            loss = loss / config.gradient_accumulation_steps
            accelerator.backward(loss)  # calculate loss gradients, loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if step % config.gradient_accumulation_steps == 0:
                optimizer.step()  # update model parameters
                lr_scheduler.step()  # update learning rate
                optimizer.zero_grad()  # reset gradients

            train_loss += loss.item()
            tokens_seen += input_ids.numel()
            if global_step < 10:  # at the beginning, log some train losses
                logger.info(f"Epoch {epoch}, step {step}, loss {loss.item()}")

            if global_step % config.eval_steps == 0:
                val_loss, perplexity = evaluate(
                    model, val_loader, accelerator=accelerator
                )
                logger.info(
                    f"Epoch {epoch} (Step {global_step:06d}): validation loss {val_loss:.3f}"
                )
                if (global_step == 0) or (val_loss < min(val_losses)):
                    best_val_loss = min(val_losses) if len(val_losses) else val_loss
                    accelerator.save_state(output_dir)
                    # TODO: keep only the k best checkpoints
                val_losses.append(val_loss)

            global_step += 1

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if global_step % config.eval_steps == 0:
                val_loss, perplexity = evaluate(model, val_loader, accelerator=accelerator)
                logger.info(
                    f"Epoch {epoch+1} (Step {global_step:06d}): validation loss {val_loss:.3f}"
                )
                if (global_step == 1) or (val_loss < min(val_losses)):
                    best_val_loss = min(val_losses) if len(val_losses) else val_loss
                    accelerator.save_state(output_dir)
                val_losses.append(val_loss)

        avg_train_loss = train_loss / len(train_loader)
        val_loss, perplexity = evaluate(model, val_loader, accelerator=accelerator)
        logger.info(f"Epoch {epoch+1}, train loss, validation loss: {avg_train_loss}, {val_loss}")
        # log validation loss and metric to wandb after each epoch
        wandb.log({"perplexity": perplexity, "val_loss": val_loss, "epoch": epoch})

        # save checkpoint at end of each epoch
        accelerator.save_state(output_dir)

        # After each epoch, print a sample text
        # checkpoint = torch.load(output_dir / "ckpt_best.pt")
        # from llamlam.model import GPTModel
        # model = GPTModel(checkpoint["model_config"])
        # model.load_state_dict(checkpoint["model"])
        # model.eval()
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info(model.generate(tokenizer, prompt="Once upon a time"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.end_training()
    wandb.finish()
