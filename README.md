# llamlam

Experiments with small language models. Two interchangeable decoder-only
architectures are implemented from scratch in PyTorch:

- **`gpt`** — a GPT-2-style transformer (`llamlam/model.py`): multi-head causal
  self-attention, GELU MLP, LayerNorm, learned positional embeddings, optional
  weight tying.
- **`diff`** — a Differential Transformer (`llamlam/difftransformer.py`,
  [Ye et al., 2024](https://arxiv.org/abs/2410.05258)): differential attention,
  SwiGLU MLP, RMSNorm.

Training uses HuggingFace `accelerate`; a DeepSpeed entry point is also provided.

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -U pip
uv pip install -r requirements.txt
pre-commit install
```

## Usage

```bash
# DiffTransformer (default) on the configured dataset
python -m llamlam.train

# GPTModel instead, with a different optimizer and bf16
python -m llamlam.train --model_type gpt --optimizer adamw --mixed_precision bf16
```

Key flags (see `python -m llamlam.train --help` and `llamlam/config.py`):

- `--model_type {diff,gpt}` — architecture to train.
- `--optimizer {adamw,muon,grokadamw}` — optimizer (`llamlam/opt.py`).
- `--mixed_precision {no,fp16,bf16}` — passed to `Accelerator`.

### DeepSpeed (multi-GPU)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
deepspeed --num_gpus $(nvidia-smi -L | wc -l) -m llamlam.train_deepspeed \
    --batch_size 16 --learning_rate 1e-4 --run_name test
```

### Muon

Muon orthogonalizes 2D weight updates via a Newton–Schulz iteration and is best
suited to GPU + larger batch sizes; biases, norms, embeddings and the (tied)
head fall back to its internal AdamW:

```bash
python -m llamlam.train --optimizer muon --learning_rate 0.02 --batch_size 32
```

## Testing

The test suite is network-free (it does not touch the HuggingFace hub):

```bash
make test                 # excludes slow tests
python -m pytest tests/    # everything, including slow overfit tests
make lint                  # ruff check + format check
```

## TeuxDeux

- Resume training from a checkpoint; keep only the k best checkpoints.
- KV cache for faster generation.
