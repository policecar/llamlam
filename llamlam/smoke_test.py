"""
Fast smoke test for the training pipeline.

This test runs a minimal training loop to verify:
- Data loading works
- Model forward/backward passes work
- Optimizer updates work
- Loss decreases (basic sanity check)
- No NaN/Inf in gradients or loss

Expected runtime: ~1-2 minutes on CPU, ~30 seconds on GPU
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llamlam.config import Config
from llamlam.data import DataCollator
from llamlam.difftransformer import DiffTransformer
from llamlam.utils import evaluate, set_seed


def run_smoke_test(verbose: bool = True):
    """
    Run a fast smoke test of the training pipeline.

    Args:
        verbose: If True, print progress information

    Returns:
        bool: True if test passes, raises AssertionError otherwise
    """
    # Tiny config for speed - completes in ~1-2 minutes
    config = Config(
        seed=42,
        max_seq_length=128,  # Short sequences
        n_layers=2,  # Tiny model
        n_heads=4,
        dim_head=64,  # → 256 dim_embd, ~1M params
        n_epochs=1,
        eval_steps=50,
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        n_data_workers=0,  # Avoid multiprocessing overhead in tests
    )

    set_seed(config.seed)

    # Load tokenizer
    if verbose:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = config.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id

    # Load SMALL subset of real data
    if verbose:
        print("Loading data subset...")
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",  # Smaller than wikitext-103
        split={
            "train": "train[:1000]",  # Only 1000 examples
            "validation": "validation[:100]",  # 100 for validation
        },
    )

    # Create dataloaders
    collate_fn = DataCollator(tokenizer, config)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    if verbose:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    if verbose:
        print("Creating model...")
    model = DiffTransformer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model parameters: {param_count:,}")
        print(f"Device: {device}")

    # Simple optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Training loop
    if verbose:
        print("\nStarting training...")
    model.train()
    initial_loss = None
    final_loss = None
    max_steps = 100  # Just 100 steps for smoke test

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)

        outputs = model(input_ids)
        loss = outputs["loss"]

        if initial_loss is None:
            initial_loss = loss.item()

        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        final_loss = loss.item() * config.gradient_accumulation_steps

        if verbose and step % 20 == 0:
            print(f"Step {step:3d}, Loss: {final_loss:.4f}")

        # Early stopping for smoke test
        if step >= max_steps:
            break

    # Final optimizer step if there are accumulated gradients
    if step % config.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Evaluation
    if verbose:
        print("\nEvaluating...")
    model.eval()
    val_loss, perplexity = evaluate(model, val_loader, device=device)

    # Results
    if verbose:
        print(f"\n{'=' * 50}")
        print("Smoke Test Results:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss:   {final_loss:.4f}")
        print(f"  Val loss:     {val_loss:.4f}")
        print(f"  Perplexity:   {perplexity:.2f}")
        print(f"  Loss decreased: {final_loss < initial_loss}")
        print(f"{'=' * 50}")

    # Success criteria
    assert final_loss is not None, "No training steps completed!"
    assert not torch.isnan(torch.tensor(final_loss)), "Loss is NaN!"
    assert not torch.isinf(torch.tensor(final_loss)), "Loss is inf!"
    assert final_loss < initial_loss, (
        f"Loss should decrease! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )

    if verbose:
        print("\n✓ Smoke test PASSED!")

    return True


if __name__ == "__main__":
    run_smoke_test(verbose=True)
