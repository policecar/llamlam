# ReadMe

Experiments with small language models.

Two decoder-only architectures, both written in PyTorch:
`diff`, a Differential Transformer (`llamlam/difftransformer.py`), and `gpt`,
a GPT-2-style transformer (`llamlam/model.py`).


## Installation

Needs Python 3.11 or later (torch 2.15.0); CI and local runs use Python 3.12.
torch 2.15.0 has wheels for macOS 14 or later on Apple silicon only (none for
Intel Macs), Linux x86_64/aarch64 with glibc 2.28 or later, and Windows x86_64.

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

# e.g. the GPT model with Muon
python -m llamlam.train --model_type gpt --optimizer muon

# resume a run where its last finished epoch left off
python -m llamlam.train --resume_from latest --output_dir <dir> --run_name <name>
```

Flags (see `python -m llamlam.train --help`; defaults live in `llamlam/config.py`):

- `--model_type {diff,gpt}`: architecture to train
- `--optimizer {adamw,muon,grokadamw}`: optimizer, see `llamlam/opt.py`
- `--learning_rate LR`: learning rate (default 6e-4); with Muon, the rate of
  its AdamW path (embeddings, head, norms, biases)
- `--muon_learning_rate LR`: Muon only (an error with other optimizers): rate
  of Muon's orthogonalised updates of the hidden weight matrices (default
  0.02); the LR schedule scales both Muon rates, and both are logged
  (`learning_rate`, `muon_learning_rate`)
- `--mixed_precision {no,fp16,bf16}`: mixed-precision mode, passed to
  `Accelerator` (accelerate 1.15.0, torch 2.15.0). Tested on Apple MPS (the
  default device on a Mac) and CPU: `bf16` autocasts the forward pass to
  bfloat16 on both; `fp16` autocasts to float16 with a gradient scaler on MPS.
  `fp16` on CPU is an error, since Accelerate would accept it and still train
  in float32; use `bf16` or `no` there. Weights and optimizer state stay
  float32. On CUDA GPUs `fp16` uses a gradient scaler and `bf16` needs
  `torch.cuda.is_bf16_supported()` (read from the accelerate source, not
  tested on a GPU)
- `--resume_from <dir>|latest`: resume from a `last/` checkpoint directory, or
  with `latest` from the run named by `--output_dir`/`--run_name`. The run
  continues with the config stored in the checkpoint, so no other flags are
  needed. Only `--n_epochs`, `--eval_steps` and `--keep_k_checkpoints` may be
  given with a different value; any other flag that differs from the stored
  value is an error. With a new `--n_epochs` the LR schedule is rebuilt for
  the new length: after the first optimizer step the rate is that of a run
  planned with that many epochs, so a decayed rate jumps back up. A lower
  `--keep_k_checkpoints` does not delete `ckpt_*` directories already on disk
  beyond the new limit; remove those by hand
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

## Parametrisation

Both models tie the input embedding to the output head (`tie_word_embeddings`,
default on). `gpt` uses the standard GPT-2-style parametrisation: attention
logits scaled by `1/sqrt(dim_head)`, weights ~ N(0, `init_std` = 0.02), residual
output projections by a further `1/sqrt(2 * n_layers)`, position embeddings zero
(`diff` keeps its Xavier init and `1/sqrt(d)` attention).

Before, `gpt` used a scheme inherited from min-max-gpt (`1/dim_head` attention,
head std `alpha / dim_embd`, embedding std `3.3 * alpha`, block std
`alpha / sqrt(dim_embd)`, `alpha = 0.5`). That was half of µP (maximal update
parametrisation; Yang & Hu, Tensor Programs IV; Yang et al., Tensor Programs V):
µP's width-dependent per-layer learning rates were never applied. Versus the
standard setting, µP scales attention logits by `1/d` instead of `1/sqrt(d)`,
shrinks the output layer's init with width and, for Adam, uses learning rates
∝ 1/width for hidden and output weights, constant for input embeddings. It lets
hyperparameters tuned on a narrow model transfer to wider ones, but costs
per-layer learning-rate groups, comparability with GPT-2-style baselines, and
sits badly with weight tying. µP is a possible future option; design notes are
in [pull request #1](https://github.com/policecar/llamlam/pull/1) (`.context/extensions/`).

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

### Two processes on a Mac

On macOS a two-process CPU run (`ACCELERATE_USE_CPU=1 python -m
torch.distributed.run --nproc_per_node 2 -m llamlam.train ...`) fails at its
first barrier with `NotImplementedError: The distributed operator
'c10d::barrier' is not currently implemented for the MPS device`: since 1.11,
accelerate passes `device_ids` to the barrier, and torch then runs the gloo
barrier on the detected accelerator, MPS. Linux and CUDA should be unaffected
(read from the source, not run). The package does not work around this;
`scripts/two_process_cpu_check.py` does, for itself only, and compares an
interrupted and resumed two-process run with an uninterrupted one (launch
commands in its docstring).

## TeuxDeux

^^ lots
