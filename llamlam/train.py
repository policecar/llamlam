import argparse
import logging
import math
import os
import yaml
import warnings

from datetime import datetime
from pathlib import Path

import torch
import wandb

from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from transformers import default_data_collator, get_scheduler

from llamlam.config import LMConfig
from llamlam.data import CustomDataset
from llamlam.model import GPTModel
from llamlam.utils import set_seed


warnings.filterwarnings("ignore", category=FutureWarning)


def eval_model(model, data_loader, device):
    """Evaluate model on a given dataset."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            loss = model(input_ids)["loss"]
            val_loss += loss.float()
        val_loss /= len(val_loader)

    try:
        perplexity = torch.exp(val_loss).item()  # type: ignore
    except OverflowError:
        perplexity = float("inf")

    return val_loss, perplexity


if __name__ == "__main__":

    ##############################
    # Setup
    ##############################

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script for LlamLam")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="experiments", help="Output directory"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument("--n_layer", type=int, default=12, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of heads")
    parser.add_argument(
        "--head_width",
        type=int,
        default=2,
        help="Width of the head, total dim is head_width * n_head",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )  # 32
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Type of learning rate scheduler",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of warmup steps"
    )
    args = parser.parse_args()

    # Load default configs
    model_config = LMConfig()
    train_config = {
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_warmup_steps": args.num_warmup_steps,
        "batch_size": args.batch_size,
        "bfloat16": {"enabled": False},  # True
        "gradient_clipping": 1.0,
    }

    # Update configs with parsed arguments
    for arg_name, arg_value in vars(args).items():
        if hasattr(model_config, arg_name):
            setattr(model_config, arg_name, arg_value)
        elif arg_name in train_config:
            train_config[arg_name] = arg_value
    logger.info(f"Arguments parsed: {vars(args)}")

    # Create run directory
    run_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(__file__).resolve().parent.parent / "data" / "runs" / run_name
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Run directory created at: {output_dir}")

    # Set seed
    seed = 137
    set_seed(seed=seed)
    logger.info(f"Using seed: {seed}")

    # Choose device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    ##############################
    # Initialize tokenizer & some
    ##############################

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = model.config.eos_token_id
    logger.info(f"Tokenizer initialized with vocabulary size: {len(tokenizer)}")

    # Update model config with tokenizer vocabulary size
    model_config.vocab_size = len(tokenizer)

    # Init wandb
    wandb.init(
        project="llamlam",
        name=run_name,
        config={**model_config.__dict__, **train_config},
    )

    # Save configs to YAML files
    model_config_path = output_dir / "model_config.yaml"
    with open(model_config_path, "w") as f:
        yaml.dump(model_config.__dict__, f)
    opt_config_path = output_dir / "train_config.yaml"
    with open(opt_config_path, "w") as f:
        yaml.dump(train_config, f)
    logger.info(f"Config saved to: {model_config_path} and {opt_config_path}")

    ##############################
    # Load data, make dataloaders
    ##############################

    train_dataset = CustomDataset(tokenizer, "train", debug=args.debug)
    val_dataset = CustomDataset(tokenizer, "validation", debug=args.debug)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=train_config["batch_size"],
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        sampler=val_sampler,
        batch_size=train_config["batch_size"] * 2,
    )

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(model_config)
    model.to(device)

    ##############################
    # Define optimizer
    ##############################

    optimizer = AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    lr_scheduler = get_scheduler(
        name=train_config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=train_config["num_warmup_steps"],
        num_training_steps=train_config["num_epochs"] * math.ceil(len(train_loader)),
    )
    logger.info(f"Optimizer initialized with config: {train_config}")

    ##############################
    # Train
    ##############################

    eval_step = 200

    val_losses = []
    tokens_seen = 0
    global_step = -1

    for epoch in range(train_config["num_epochs"]):
        train_loss = 0.0
        for batch in train_loader:

            model.train()
            optimizer.zero_grad()  # reset gradients
            input_ids = batch["input_ids"].to(device)
            loss = model(input_ids)["loss"]  # logits, (loss), (hidden_states)
            loss.backward()  # calculate loss gradients
            optimizer.step()  # update model parameters

            wandb.log(
                {"train_loss": loss.item()}
            )  # log train loss to wandb after each batch
            train_loss += loss.item()
            tokens_seen += input_ids.numel()
            global_step += 1

            if global_step % eval_step == 0:
                val_loss, perplexity = eval_model(model, val_loader, device)
                logger.info(
                    f"Epoch {epoch+1} (Step {global_step:06d}): validation loss {val_loss:.3f}"
                )
                if not len(val_losses) or (val_loss < min(val_losses)):
                    model.save_pretrained(output_dir, "best", optimizer)
                    logger.info(f"Saved model checkpoint at step {global_step}")
                val_losses.append(val_loss)

        avg_train_loss = train_loss / len(train_loader)
        val_loss, perplexity = eval_model(model, val_loader, device)
        logger.info(
            f"Epoch {epoch+1}, train loss, validation loss: {avg_train_loss}, {val_loss}"
        )
        # log validation loss and metric to wandb after each epoch
        wandb.log({"perplexity": perplexity, "val_loss": val_loss, "epoch": epoch})

        # save model at end of each epoch
        model.save_pretrained(
            output_dir, f"epoch_{epoch}_step_{global_step}", optimizer
        )

        # After each epoch, print a sample text
        # model = GPTModel.from_pretrained(
        #     model_config, {"model_name_or_path": output_dir / "model_best.pt"}
        # )
        logger.info(model.generate(tokenizer, prompt="Once upon a time"))

    wandb.finish()
