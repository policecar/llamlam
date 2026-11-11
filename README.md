# ReadMe

Experiments with small language models.

Two decoder-only architectures, both written in PyTorch:
`diff`, a Differential Transformer (`llamlam/difftransformer.py`), and `gpt`,
a GPT-2-style transformer (`llamlam/model.py`).


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

```bash
python -m llamlam.train

# e.g. the GPT model with Muon and bf16
python -m llamlam.train --model_type gpt --optimizer muon --mixed_precision bf16
```

Flags (see `python -m llamlam.train --help`; defaults live in `llamlam/config.py`):

- `--model_type {diff,gpt}`: architecture to train
- `--optimizer {adamw,muon,grokadamw}`: optimizer, see `llamlam/opt.py`
- `--mixed_precision {no,fp16,bf16}`: mixed-precision mode, passed to `Accelerator`
- `--resume_from <dir>|latest`: resume from a checkpoint directory, or with
  `latest` from the run named by `--output_dir`/`--run_name`
- `--keep_k_checkpoints N`: number of best (lowest validation loss) checkpoints to keep
- `--n_layers`, `--n_heads`, `--dim_head`: model size (embedding size is `n_heads * dim_head`)
- `--run_name`, `--output_dir`: run directory is `<output_dir>/<run_name>`
- `--debug`: log at DEBUG level

`llamlam/train_deepspeed.py` is currently not maintained: it does not run
against the current code and has not been verified with DeepSpeed. Use
`python -m llamlam.train`.

### Generation

```python
text = model.generate(
    tokenizer, prompt, max_new_tokens=100,
    do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
)
```

Decoding is greedy unless `do_sample=True`. It uses a key-value cache while the
sequence fits in `max_seq_length`; past that, each step re-encodes the last
`max_seq_length` tokens.

## Tests and linting

```bash
make test                               # full suite, including slow tests
python -m pytest -m "not slow" tests/   # skip slow tests
make lint                               # ruff check + ruff format --check
make format                             # ruff format (llamlam/ and tests/ only)
```

Some tests load the GPT-2 tokenizer (`AutoTokenizer.from_pretrained("gpt2")`),
so they need access to the Hugging Face hub or an already populated local cache.
CI (`.github/workflows/ci.yml`) runs the same lint and test commands on every
push and pull request.

## TeuxDeux

^^ lots
