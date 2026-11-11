import argparse
import dataclasses
import logging
import math
import os
import yaml

from datetime import datetime
from pathlib import Path

import torch
import wandb

from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

from llamlam.checkpoint import CheckpointManager, load_checkpoint, resolve_resume_dir
from llamlam.config import (
    MIXED_PRECISION_MODES,
    MODEL_TYPES,
    OPTIMIZERS,
    Config,
    check_choice,
)
from llamlam.data import DataCollator, drop_short_examples, train_val_split

from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel
from llamlam.utils import build_optimizer, evaluate, set_seed


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["MALLOC_DISABLE_WARNINGS"] = "1"

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"

# CLI options that configure the run itself rather than a Config field
RUN_OPTIONS = ("debug", "output_dir", "run_name", "resume_from")

MODEL_CLASSES = {"diff": DiffTransformer, "gpt": GPTModel}  # keys: config.MODEL_TYPES


##########################################
# Arguments & config
##########################################


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Training script for LlamLam")
    parser.add_argument(
        "--debug", action="store_true", help="Log at DEBUG level instead of INFO"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        help=f"Parent directory for run directories (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Run name, also the run directory name (default: test_<timestamp>)",
    )
    parser.add_argument(
        "--n_layers", "--n_layer", dest="n_layers", type=int, help="Number of layers"
    )
    parser.add_argument(
        "--n_heads", "--n_head", dest="n_heads", type=int, help="Number of heads"
    )
    parser.add_argument(
        "--dim_head",
        type=int,
        help="Dimension of each attention head, total dim is n_heads * dim_head",
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
    parser.add_argument(
        "--model_type", choices=MODEL_TYPES, help="Model architecture (default: diff)"
    )
    parser.add_argument(
        "--optimizer", choices=OPTIMIZERS, help="Optimizer (default: adamw)"
    )
    parser.add_argument(
        "--mixed_precision",
        choices=MIXED_PRECISION_MODES,
        help="Mixed-precision mode passed to Accelerator (default: no)",
    )
    parser.add_argument(
        "--keep_k_checkpoints",
        type=int,
        help="Number of best (lowest validation loss) checkpoints to keep",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Resume from a last/ checkpoint directory, or 'latest' for the last/ "
        "of the run named by --output_dir/--run_name. Resume is epoch-granular",
    )
    return parser.parse_args(argv)


def configure_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger().setLevel(level)  # basicConfig is a no-op if already set up


def apply_overrides(config, overrides):
    """Set each override on config; raise if any name is not a Config field."""
    field_names = {f.name for f in dataclasses.fields(config)}
    unknown = sorted(set(overrides) - field_names)
    if unknown:
        raise ValueError(
            f"Cannot apply config override(s) {unknown}: not Config fields "
            f"(valid: {sorted(field_names)})"
        )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def build_config(args, config=None):
    """Default (or given) Config with every non-None CLI value applied."""
    config = Config() if config is None else config
    overrides = {
        name: value
        for name, value in vars(args).items()
        if name not in RUN_OPTIONS and value is not None
    }
    return apply_overrides(config, overrides)


def resolve_run_dir(args):
    """
    Return (run_name, run_dir) from --run_name / --output_dir.

    With --resume_from <checkpoint dir> and neither flag given, the run continues
    in the directory that holds the checkpoint; given flags always win.
    """
    resume_from = getattr(args, "resume_from", None)
    if resume_from not in (None, "latest") and not (args.run_name or args.output_dir):
        run_dir = Path(resume_from).resolve().parent
        return run_dir.name, run_dir
    run_name = args.run_name or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parent = Path(args.output_dir) if args.output_dir else DEFAULT_RUNS_DIR
    return run_name, parent / run_name


##########################################
# Model & accelerator
##########################################


def build_model(config):
    """The model named by config.model_type; raises ValueError for unknown names."""
    return MODEL_CLASSES[check_choice("model_type", config.model_type, MODEL_TYPES)](
        config
    )


def build_accelerator(config, **kwargs):
    """Accelerator with config.mixed_precision; raises ValueError for unknown modes."""
    mixed_precision = check_choice(
        "mixed_precision", config.mixed_precision, MIXED_PRECISION_MODES
    )
    return Accelerator(mixed_precision=mixed_precision, **kwargs)


##########################################
# Gradient accumulation
##########################################

# @decision DEC-001: an optimizer step after every `accum` micro-batches, plus one
#   for a trailing partial group at the end of each epoch
# @rationale: `step % accum == 0` stepped after the first micro-batch and left the
#   epoch's last gradients to leak into the next epoch's first step. The trailing
#   group's loss is divided by its real size so every step averages its micro-batches.


def is_accumulation_boundary(step, n_micro_batches, accum):
    """True if the optimizer should step after micro-batch `step` (0-based)."""
    return (step + 1) % accum == 0 or step + 1 == n_micro_batches


def accumulation_group_size(step, n_micro_batches, accum):
    """Number of micro-batches in the accumulation group containing `step`."""
    group_start = (step // accum) * accum
    return min(accum, n_micro_batches - group_start)


def num_training_steps(n_epochs, n_micro_batches, accum):
    """Optimizer steps taken over training (each epoch flushes a partial group)."""
    return n_epochs * math.ceil(n_micro_batches / accum)


# @decision DEC-002: build the LR scheduler after accelerator.prepare, from the
#   sharded train loader's length, with warmup and total multiplied by
#   num_processes (unless split_batches)
# @rationale: accelerate 1.0.1's AcceleratedScheduler.step (scheduler.py) advances
#   the wrapped scheduler num_processes times per call when split_batches is False,
#   and once when it is True. Each process takes num_training_steps(...) optimizer
#   steps over its own shard, so scaling both counts by that factor makes the
#   schedule, measured in optimizer steps, identical to the single-process one and
#   ends it exactly on the last optimizer step. Assumes step_scheduler_with_optimizer
#   (the Accelerator default) and no accelerator.accumulate (sync_gradients stays
#   True, so every step() call advances the schedule).


def lr_schedule_steps(
    n_epochs, n_micro_batches, accum, n_warmup_steps, num_processes, split_batches
):
    """
    (num_warmup_steps, num_training_steps) to pass to get_scheduler so that, once
    wrapped by accelerator.prepare, the schedule warms up over n_warmup_steps
    optimizer steps and ends on the last one.

    n_micro_batches is the length of the *prepared* (per-process) train loader.
    """
    if num_processes < 1:
        raise ValueError(f"num_processes must be >= 1, got {num_processes}")
    steps_per_call = 1 if split_batches else num_processes
    n_optimizer_steps = num_training_steps(n_epochs, n_micro_batches, accum)
    return n_warmup_steps * steps_per_call, n_optimizer_steps * steps_per_call


##########################################
# Checkpoints & sampling
##########################################


def count_real_tokens(batch):
    """Non-padding tokens in a collated batch (all of them without a mask)."""
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return batch["input_ids"].numel()
    return int(attention_mask.sum().item())


def sum_over_processes(accelerator, value):
    """Sum an int over all processes (a collective: every process must call it)."""
    total = torch.tensor(value, dtype=torch.int64, device=accelerator.device)
    return int(accelerator.reduce(total, reduction="sum").item())


def sample_text(accelerator, model, tokenizer, prompt="Once upon a time"):
    """Generate from the underlying model (the prepared one may be wrapped)."""
    return accelerator.unwrap_model(model).generate(tokenizer, prompt=prompt)


##########################################
# Training loop
##########################################


def run_training(
    model,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    accelerator,
    config,
    run_dir,
    tokenizer=None,
    log=None,
    resume_dir=None,
):
    """
    Train for config.n_epochs. Objects are expected to be accelerator-prepared,
    the LR scheduler included.

    After every epoch the state is saved to `run_dir/last`; every evaluation
    whose loss is among the config.keep_k_checkpoints lowest is kept as
    `run_dir/ckpt_<global_step>` (see llamlam.checkpoint).

    `resume_dir`, if given, is a last/ checkpoint: model, optimizer, LR scheduler
    and RNG state and the counters are restored from it and training continues
    with the epoch after the one it saved. Resume is epoch-granular: an epoch
    interrupted part-way is trained again from its start. When resuming in the
    checkpoint's own run directory, the kept-best list carries over.

    `log(metrics, step=None)` receives metrics (e.g. wandb.log); `tokenizer`, if
    given, is used to print a sample after each epoch.

    Counters: global_step counts micro-batches trained, tokens_seen non-padding
    tokens over all processes. Returns a dict with best_val_loss,
    n_optimizer_steps, global_step, tokens_seen (all over the whole run, resumed
    part included) and avg_train_loss (of the last epoch trained, nan if none).
    """
    log = log or (lambda *args, **kwargs: None)
    accum = config.gradient_accumulation_steps
    n_micro_batches = len(train_loader)
    checkpoints = CheckpointManager(
        accelerator, run_dir, keep_k=config.keep_k_checkpoints
    )

    start_epoch = 0
    global_step = 0
    n_optimizer_steps = 0
    tokens_seen = 0
    best_val_loss = math.inf
    avg_train_loss = float("nan")

    if resume_dir is not None:
        progress = load_checkpoint(accelerator, resume_dir)
        start_epoch = progress["epoch"]
        global_step = progress["global_step"]
        n_optimizer_steps = progress["n_optimizer_steps"]
        tokens_seen = progress["tokens_seen"]
        best_val_loss = progress["best_val_loss"]
        if Path(resume_dir).resolve().parent == Path(run_dir).resolve():
            checkpoints.restore(progress)
        logger.info(
            f"Resumed from {resume_dir}: starting epoch {start_epoch}, step "
            f"{global_step}, {n_optimizer_steps} optimizer steps, LR "
            f"{lr_scheduler.get_last_lr()[0]:.3e}, kept best {checkpoints.best}"
        )
        if start_epoch >= config.n_epochs:
            logger.warning(
                f"Checkpoint is at epoch {start_epoch}, n_epochs is {config.n_epochs}: "
                f"nothing left to train"
            )

    def progress_at(epoch, global_step, tokens_seen):
        return {
            "epoch": epoch,
            "global_step": global_step,
            "n_optimizer_steps": n_optimizer_steps,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
        }

    for epoch in range(start_epoch, config.n_epochs):
        train_loss = 0.0
        epoch_tokens = 0  # this process's share, summed over processes when used
        optimizer.zero_grad()  # reset gradients
        for step, batch in enumerate(train_loader):
            model.train()
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            loss = model(input_ids, attention_mask=attention_mask)["loss"]
            # scale for accumulation only in backward; log the unscaled loss
            group_size = accumulation_group_size(step, n_micro_batches, accum)
            accelerator.backward(loss / group_size)

            if is_accumulation_boundary(step, n_micro_batches, accum):
                if config.gradient_clipping:
                    accelerator.clip_grad_norm_(
                        model.parameters(), max_norm=config.gradient_clipping
                    )
                optimizer.step()  # update model parameters
                lr_scheduler.step()  # update learning rate
                optimizer.zero_grad()  # reset gradients
                n_optimizer_steps += 1

            train_loss += loss.item()
            epoch_tokens += count_real_tokens(batch)
            if global_step < 10:  # at the beginning, log some train losses
                logger.info(f"Epoch {epoch}, step {step}, loss {loss.item()}")

            if global_step % config.eval_steps == 0:
                val_loss, perplexity = evaluate(
                    model, val_loader, accelerator=accelerator
                )
                logger.info(
                    f"Epoch {epoch} (Step {global_step:06d}): validation loss {val_loss:.3f}"
                )
                best_val_loss = min(best_val_loss, val_loss)
                # a mid-epoch snapshot, after global_step + 1 micro-batches
                tokens_now = tokens_seen + sum_over_processes(accelerator, epoch_tokens)
                checkpoints.save_best(
                    val_loss, progress_at(epoch, global_step + 1, tokens_now)
                )

            global_step += 1

            log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                },
                step=global_step,
            )

        tokens_seen += sum_over_processes(accelerator, epoch_tokens)
        avg_train_loss = train_loss / n_micro_batches
        val_loss, perplexity = evaluate(model, val_loader, accelerator=accelerator)
        logger.info(
            f"Epoch {epoch+1}, train loss, validation loss: {avg_train_loss}, {val_loss}"
        )
        # log validation loss and metric after each epoch
        log(
            {
                "perplexity": perplexity,
                "val_loss": val_loss,
                "tokens_seen": tokens_seen,
                "epoch": epoch,
            }
        )

        best_val_loss = min(best_val_loss, val_loss)
        checkpoints.save_best(val_loss, progress_at(epoch, global_step, tokens_seen))
        # resumable: a resume starts with the next epoch
        checkpoints.save_last(progress_at(epoch + 1, global_step, tokens_seen))

        # After each epoch, print a sample text
        if tokenizer is not None and accelerator.is_main_process:
            logger.info(sample_text(accelerator, model, tokenizer))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "best_val_loss": best_val_loss,
        "n_optimizer_steps": n_optimizer_steps,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "avg_train_loss": avg_train_loss,
    }


def main(argv=None):
    ##########################################
    # Setup
    ##########################################

    args = parse_args(argv)
    configure_logging(debug=args.debug)
    config = build_config(args)
    logger.info(f"Arguments parsed: {vars(args)}")

    # Resolve the run and resume directories (raises before anything is created)
    run_name, output_dir = resolve_run_dir(args)
    resume_dir = resolve_resume_dir(output_dir, args.resume_from)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Run directory: {output_dir}")
    if resume_dir is not None:
        logger.info(f"Resuming from: {resume_dir}")

    # Set seed
    set_seed(seed=config.seed)
    logger.info(f"Using seed: {config.seed}")

    accelerator = build_accelerator(config)

    ##########################################
    # Initialize tokenizer & some
    ##########################################

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = config.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer initialized with vocabulary size: {len(tokenizer)}")

    # Update model config with tokenizer vocabulary size
    config.vocab_size = len(tokenizer)

    ##########################################
    # Init WandB, save config
    ##########################################

    wandb.init(
        project="llamlam",
        name=run_name,
        config=config.to_dict(),
    )

    # Save config to YAML file
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f)
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
        raise ValueError(f"Failed to load dataset: {e}") from e

    # Use the dataset's validation split, or split one off the training set
    train_set, val_set = train_val_split(dataset)

    # Drop all columns except 'text'
    train_set = train_set.select_columns(["text"])
    val_set = val_set.select_columns(["text"])

    # Drop examples without a next-token target (< 2 tokens); the main process
    # writes the datasets cache first, the others then reuse it
    with accelerator.main_process_first():
        train_set = drop_short_examples(
            train_set, tokenizer, config.max_seq_length, name="train"
        )
        val_set = drop_short_examples(
            val_set, tokenizer, config.max_seq_length, name="validation"
        )

    # Define collate function
    collate_fn = DataCollator(tokenizer, config)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.n_data_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.n_data_workers,
        pin_memory=True,
    )

    ##########################################
    # Instantiate model
    ##########################################

    model = build_model(config)
    logger.info(f"Model: {type(model).__name__} (model_type={config.model_type})")

    ##########################################
    # Define optimizer
    ##########################################

    optimizer = build_optimizer(model, config)

    logger.info(
        f"Optimizer {type(optimizer).__name__} initialized with config: {config}"
    )

    ##########################################
    # Prepare, then build the LR scheduler
    ##########################################

    # ~"no specific order, we just need to unpack objects in the same order we gave them to the prepare method"
    base_optimizer = optimizer
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Counts come from the sharded loader; see lr_schedule_steps
    accum = config.gradient_accumulation_steps
    n_warmup_steps, n_scheduler_steps = lr_schedule_steps(
        config.n_epochs,
        len(train_loader),
        accum,
        config.n_warmup_steps,
        accelerator.num_processes,
        accelerator.split_batches,
    )
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=base_optimizer,  # prepare_scheduler matches it to its wrapper
        num_warmup_steps=n_warmup_steps,
        num_training_steps=n_scheduler_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)
    n_optimizer_steps = num_training_steps(config.n_epochs, len(train_loader), accum)
    logger.info(
        f"LR scheduler: {len(train_loader)} micro-batches per process per epoch, "
        f"{n_optimizer_steps} optimizer steps, scheduler warmup/total "
        f"{n_warmup_steps}/{n_scheduler_steps} over {accelerator.num_processes} "
        f"process(es)"
    )

    ##########################################
    # Training loop
    ##########################################

    # The scheduler is prepared above, so resuming (accelerator.load_state inside
    # run_training) restores its position along with the model and optimizer
    run_training(
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        accelerator,
        config,
        output_dir,
        tokenizer=tokenizer,
        log=wandb.log,
        resume_dir=resume_dir,
    )

    accelerator.end_training()
    wandb.finish()


if __name__ == "__main__":
    main()
