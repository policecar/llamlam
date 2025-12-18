import argparse
import json
import logging
import os
import yaml

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import wandb

from accelerate import Accelerator
from datasets import load_dataset
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

from llamlam.config import Config
from llamlam.data import DataCollator

from llamlam.difftransformer import DiffTransformer
from llamlam.utils import evaluate, get_grouped_params, set_seed


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MALLOC_DISABLE_WARNINGS"] = "1"


def cleanup_old_checkpoints(output_dir: Path, keep_best_k: int = 3) -> None:
    """Keep only the k best checkpoints based on validation loss.

    Args:
        output_dir: Directory containing checkpoints and checkpoint_tracker.json
        keep_best_k: Number of best checkpoints to keep
    """
    tracker_file = output_dir / "checkpoint_tracker.json"
    if not tracker_file.exists():
        return

    with open(tracker_file, "r") as f:
        checkpoints = json.load(f)

    # Sort by validation loss (ascending)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x["val_loss"])

    # Keep only the best k
    checkpoints_to_keep = sorted_checkpoints[:keep_best_k]
    checkpoints_to_delete = sorted_checkpoints[keep_best_k:]

    # Delete old checkpoints
    for ckpt in checkpoints_to_delete:
        ckpt_dir = Path(ckpt["path"])
        if ckpt_dir.exists() and ckpt_dir != output_dir:
            # Delete the checkpoint directory
            import shutil
            shutil.rmtree(ckpt_dir)
            logging.info(f"Deleted checkpoint: {ckpt_dir} (val_loss: {ckpt['val_loss']:.4f})")

    # Update tracker file
    with open(tracker_file, "w") as f:
        json.dump(checkpoints_to_keep, f, indent=2)


def save_checkpoint_with_tracking(
    accelerator: Accelerator,
    output_dir: Path,
    global_step: int,
    val_loss: float,
    keep_best_k: int = 3,
) -> None:
    """Save checkpoint and track it for cleanup.

    Args:
        accelerator: Accelerator instance for saving
        output_dir: Base output directory
        global_step: Current global step
        val_loss: Validation loss for this checkpoint
        keep_best_k: Number of best checkpoints to keep
    """
    # Create checkpoint directory
    ckpt_dir = output_dir / f"checkpoint-{global_step}"
    accelerator.save_state(ckpt_dir)

    # Update checkpoint tracker
    tracker_file = output_dir / "checkpoint_tracker.json"
    checkpoints = []
    if tracker_file.exists():
        with open(tracker_file, "r") as f:
            checkpoints = json.load(f)

    # Add new checkpoint
    checkpoints.append({
        "path": str(ckpt_dir),
        "global_step": global_step,
        "val_loss": val_loss,
        "timestamp": datetime.now().isoformat(),
    })

    # Save updated tracker
    with open(tracker_file, "w") as f:
        json.dump(checkpoints, f, indent=2)

    # Cleanup old checkpoints
    cleanup_old_checkpoints(output_dir, keep_best_k)


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: int,
    prefix: str = "",
) -> None:
    """Log metrics to both logger and wandb.

    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
        step: Current step for wandb logging
        prefix: Optional prefix for log messages
    """
    # Log to console
    metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                             for k, v in metrics.items()])
    if prefix:
        logger.info(f"{prefix} - {metrics_str}")
    else:
        logger.info(metrics_str)

    # Log to wandb
    wandb.log(metrics, step=step)


if __name__ == "__main__":
    ##########################################
    # Setup
    ##########################################

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training script for LlamLam")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="experiments", help="Output directory"
    )
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint directory to resume from"
    )
    parser.add_argument(
        "--keep_best_k", type=int, default=3, help="Number of best checkpoints to keep"
    )
    parser.add_argument("--n_layer", type=int, help="Number of layers")
    parser.add_argument("--n_head", type=int, help="Number of heads")
    parser.add_argument(
        "--dim_head",
        type=int,
        help="Dimension of each attention head, total dim is n_head * dim_head",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, help="Number of training epochs")
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
    parser.add_argument("--n_warmup_steps", type=int, help="Number of warmup steps")
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
    tokenizer.model_max_length = config.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer initialized with vocabulary size: {len(tokenizer)}")

    # Update model config with tokenizer vocabulary size and pad_token_id
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id

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
        num_workers=config.n_data_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset["test"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.n_data_workers,
        pin_memory=True,
    )

    ##########################################
    # Instantiate model
    ##########################################

    model = DiffTransformer(config)
    # model = GPTModel(config)
    # model.to(device)

    ##########################################
    # Define optimizer
    ##########################################

    optimizer = AdamW(
        get_grouped_params(
            model, weight_decay=config.weight_decay, no_decay=config.no_decay
        ),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.n_warmup_steps,
        num_training_steps=config.n_epochs
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
    global_step = 0
    starting_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        accelerator.load_state(args.resume)

        # Load training state metadata
        state_file = Path(args.resume) / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                training_state = json.load(f)
                global_step = training_state["global_step"]
                starting_epoch = training_state["epoch"]
                val_losses = training_state["val_losses"]
                logger.info(
                    f"Resumed from epoch {starting_epoch}, step {global_step}, "
                    f"best val loss: {min(val_losses) if val_losses else 'N/A'}"
                )
        else:
            logger.warning("training_state.json not found, starting from scratch")

    try:
        for epoch in range(starting_epoch, config.n_epochs):
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

                if (step + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()  # update model parameters
                    lr_scheduler.step()  # update learning rate
                    optimizer.zero_grad()  # reset gradients

                train_loss += loss.item()
                if global_step < 10:  # at the beginning, log some train losses
                    logger.info(f"Epoch {epoch}, step {step}, loss {loss.item()}")

                if global_step % config.eval_steps == 0:
                    val_loss, perplexity = evaluate(
                        model, val_loader, accelerator=accelerator
                    )

                    # Save checkpoint with tracking and cleanup
                    if len(val_losses) == 0 or val_loss < min(val_losses):
                        logger.info(f"New best validation loss: {val_loss:.3f}, saving checkpoint")

                    save_checkpoint_with_tracking(
                        accelerator, output_dir, global_step, val_loss, args.keep_best_k
                    )

                    # Save training state metadata
                    state_file = output_dir / f"checkpoint-{global_step}" / "training_state.json"
                    with open(state_file, "w") as f:
                        json.dump({
                            "global_step": global_step,
                            "epoch": epoch,
                            "val_losses": val_losses + [val_loss],
                        }, f, indent=2)

                    val_losses.append(val_loss)

                    # Log validation metrics
                    log_metrics(
                        logger,
                        {"val_loss": val_loss, "perplexity": perplexity},
                        global_step,
                        prefix=f"Epoch {epoch} (Step {global_step:06d})"
                    )

                global_step += 1

                # Log training metrics
                log_metrics(
                    logger,
                    {
                        "train_loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                    prefix=""
                )

            avg_train_loss = train_loss / len(train_loader)
            val_loss, perplexity = evaluate(model, val_loader, accelerator=accelerator)

            # Log end-of-epoch metrics
            log_metrics(
                logger,
                {
                    "avg_train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "perplexity": perplexity,
                    "epoch": epoch,
                },
                step=global_step,
                prefix=f"Epoch {epoch+1} complete"
            )

            # Save checkpoint at end of each epoch
            save_checkpoint_with_tracking(
                accelerator, output_dir, global_step, val_loss, args.keep_best_k
            )

            # Save training state metadata
            state_file = output_dir / f"checkpoint-{global_step}" / "training_state.json"
            with open(state_file, "w") as f:
                json.dump({
                    "global_step": global_step,
                    "epoch": epoch + 1,  # Next epoch to start from
                    "val_losses": val_losses,
                }, f, indent=2)

            val_losses.append(val_loss)

            # After each epoch, print a sample text
            logger.info(model.generate(tokenizer, prompt="Once upon a time"))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info("Saving checkpoint before exit...")
        try:
            save_checkpoint_with_tracking(
                accelerator, output_dir, global_step, val_losses[-1] if val_losses else float('inf'), args.keep_best_k
            )
            logger.info("Checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA out of memory error!")
            logger.error("Try reducing batch size or sequence length")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.error(f"Runtime error during training: {e}")
            raise

    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.end_training()
        wandb.finish()
        logger.info("Training session ended")
