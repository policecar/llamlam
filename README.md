# ReadMe

Experiments with small language models.


## Installation

```bash
# Make virtual env
uv venv
source .venv/bin/activate

# Install Python packages
uv pip install -U pip
uv pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

## Usage

### Training

```bash
python -m llamlam.train
```

### Quick Smoke Test

Run a fast smoke test (~1-2 minutes) to verify the training pipeline works:

```bash
make smoke-test
# or
python -m llamlam.smoke_test
```

This runs a minimal training loop with:
- Tiny 2-layer model (~1M parameters)
- 1000 training samples from WikiText-2
- 100 training steps
- Verifies loss decreases and no NaN/Inf values

Use this to quickly test changes before running full training.

## Development

```bash
# Run tests
make test

# Run smoke test
make smoke-test

# Format code
make format

# Lint code
make lint
```

## TeuxDeux

^^ lots
