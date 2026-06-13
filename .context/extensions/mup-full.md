# Extension: Maximal Update Parametrization (μP) — full version

Status: **design / implementation guide** (not yet implemented).
Audience: a contributor adding `parametrization="mup"` to llamlam, done *properly*
(Tensor-Programs μP with base shapes + coordinate check), not the single
`width_mult` shortcut.

> Want the quick version instead? See [`mup-lightweight.md`](./mup-lightweight.md).
> Not sure which? Read the [index](./README.md).

---

## 1. Why μP, and why the *full* version

Under standard parametrization (SP) the optimal learning rate (and init, and
multipliers) drift as you change width, so HPs tuned small don't transfer big.
**μP** (Yang & Hu, *Tensor Programs V*, 2022) fixes the per-layer init scale,
learning rate, and forward multipliers so the optimal HPs become
**width-invariant** → tune on a tiny model, transfer to a large one (*μTransfer*).

The **lightweight** recipe (see git history of this file) collapses everything
to one global `width_mult = dim_embd / base`. That is only correct when *every*
infinite dimension scales together by the same factor and every hidden matrix is
square in width. It silently breaks the moment you:

- scale `dim_head` and `n_heads` independently (so attention `fan_in` ≠ embed width),
- change the MLP expansion ratio (the `4*dim_embd` hidden is a non-square map),
- want the readout's true `Θ(1/fan_in)` scaling rather than `Θ(1/width_mult)`,
- scale depth and width together.

The **full** version tracks a *per-parameter, per-dimension* width multiplier
derived from a **base-shapes** comparison, so each weight gets the right init /
LR / multiplier regardless of how you scale. That machinery (and the coordinate
check that verifies it) is the whole point of "full μP".

---

## 2. What's already in this repo

`llamlam` was cribbed from [min-max-gpt](https://github.com/cloneofsimo/min-max-gpt),
a μP repo, and fragments survived into the pre-refactor code:

- `model.py` attention used `scale = 1/dim_head` — the μP **1/d attention**
  (SP wants `1/sqrt(dim_head)`).
- Embedding init `std ≈ 1.65` — a width-independent **Θ(1)** input init.
- Head init `std ≈ 1/(2·dim_embd)` — the μP **Θ(1/fan_in)** readout.
- `utils.get_grouped_params` had commented-out per-group LR:
  `lr * (3.3 if is_embed else 1.0)` and `lr * (1/dim_head)`.

So ~80% of a μP was there; the missing pieces were the per-group LR coupling and
the readout output multiplier. The refactor reset the default to clean SP
(`init_std=0.02`, `1/sqrt(d)` attention, weight tying). This doc adds μP back as
an explicit, **base-shape-correct**, tested option.

---

## 3. The μP formalism (abc-parametrization)

Every weight `W_l` is written with three exponents over the width `n`:

```
W_l = n^{-a_l} · w_l ,   w_l ~ N(0, n^{-2 b_l} · σ²) ,   Adam LR  η_l = η · n^{-c_l}
```

μP chooses `(a_l, b_l, c_l)` per *layer class* so that, in the infinite-width
limit, **every layer's pre-activations and their updates are Θ(1)** ("maximal"
feature learning). Classes are decided by which of `fan_in`/`fan_out` are
*infinite* (scale with width) vs *finite* (fixed: vocab, seq len, 1 for biases).

### μP table for **Adam** (the optimizer llamlam uses)

Let `n` = the *infinite fan-in* of the layer; `m = n / n_base` its width ratio.

| Layer class (fan_in → fan_out) | Init std | Forward multiplier | Adam LR |
|---|---|---|---|
| **Input** (finite → ∞): token & pos embeddings, biases | `σ` (const) | `Θ(1)` (`input_mult`) | `η` |
| **Hidden** (∞ → ∞): qkv, attn out_proj, MLP in/out | `σ / √m` | `1` | `η / m` |
| **Output** (∞ → finite): readout/head | `σ / √m` *(or 0)* | `output_mult / m` | `η / m` |
| **Vector** (gains): LayerNorm/RMSNorm `weight` | `1` | `1` | `η` |
| **Attention scores** | — | scale by **`1/dim_head`** (not `1/√dim_head`) | — |

`input_mult`, `output_mult` are O(1) tunables (default 1; the old code's `3.3`
was an `input_mult`). Common extras: **zero-init the readout** and **zero-init
the query projection** so logits/attention start at 0 (stabilizes early steps).

> Full μP computes `m` **per parameter** from the base shapes (Section 4), and
> the **Output** row uses the *actual* `fan_in` ratio (e.g. for the MLP
> down-projection `4d → d`, `fan_in = 4d`). The lightweight recipe wrongly reuses
> one global `m` everywhere.

---

## 4. Base shapes — the core mechanism

μP needs to know, per parameter dimension, *which dims are infinite and their
base size*. You supply this by instantiating the model at the real shape plus a
**base** model (infinite dims set to a small base, e.g. `dim_embd=256`,
`dim_head=32`, `n_heads` fixed) and a **delta** model (base with the infinite
dims bumped, e.g. `+ a different value`) so the tooling can *disambiguate* which
dims move. Comparing the three assigns every parameter an `infshape`
(per-dim `(current, base)` ratio). `width_mult()` for a parameter is the product
of its *infinite fan-in* ratios.

This is exactly what `mup.set_base_shapes(model, base_model, delta_model)` does.
Doing it by hand means writing a small `InfShape` tracker (Section 6, Path B).

**Checkpoint note:** base shapes are *not* in `state_dict`. Save them
(`mup.save_base_shapes` / your own) and re-apply on load, or fold all μP
multipliers into the weights at export so inference is parametrization-agnostic.

---

## 5. Path A — use the `mup` library (recommended for correctness)

```bash
uv pip install mup    # verify torch compatibility; mup is older, may need a pin
```

### 5.1 Config (`llamlam/config.py`)

```python
parametrization: str = "sp"          # "sp" or "mup"
mup_base_dim_embd: int = 256         # base infinite width for dim_embd
mup_base_dim_head: int = 32          # base infinite width for dim_head
mup_input_mult: float = 1.0
mup_output_mult: float = 1.0
mup_readout_zero_init: bool = True
mup_query_zero_init: bool = True
```

### 5.2 Model edits (`llamlam/model.py`)

Use `MuReadout` for the head, `1/dim_head` attention, and apply the input mult:

```python
from mup import MuReadout, normal_

class GPTModel(nn.Module):
    def __init__(self, config):
        ...
        self.head = MuReadout(
            config.dim_embd, config.vocab_size, bias=False,
            output_mult=config.mup_output_mult,
            readout_zero_init=config.mup_readout_zero_init,
        ) if config.parametrization == "mup" else nn.Linear(...)
        # NB: do NOT tie weights under μP — embedding (Input) and readout
        # (Output) live in different μP classes.

    def forward(self, input_ids, attention_mask=None, labels=None, ...):
        x = self.embed(input_ids) * self.input_mult + self.pos_embed[:, :T, :]
        ...
        logits = self.head(x).float()   # MuReadout applies output_mult / width_mult
```

Attention scale becomes μP-aware (pass into `Context`, then to SDPA):

```python
# Context.__init__: self.attn_scale = (1.0/dim_head) if mup else None
attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.attn_scale)
```

`_init_weights` should use `mup.normal_` (base-shape-aware) instead of
`nn.init.normal_`, and optionally zero the query projection:

```python
def _init_weights(self, module):
    if isinstance(module, (nn.Linear, MuReadout)):
        normal_(module.weight, mean=0.0, std=self.config.init_std)  # mup.normal_
        if module.bias is not None: nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        normal_(module.weight, mean=0.0, std=self.config.init_std)
# after build: if mup_query_zero_init, zero the q slice of each qkv_proj.weight
# drop the SP residual 1/sqrt(2L) rescale under μP — the per-layer LR handles it.
```

### 5.3 Base shapes at construction (`train.py` / a factory)

```python
from mup import set_base_shapes

def build_model(config):
    model = GPTModel(config)
    if config.parametrization == "mup":
        base  = GPTModel(replace(config, dim_embd=config.mup_base_dim_embd,
                                 dim_head=config.mup_base_dim_head))
        delta = GPTModel(replace(config, dim_embd=config.mup_base_dim_embd*2,
                                 dim_head=config.mup_base_dim_head*2))
        set_base_shapes(model, base, delta)   # attaches infshape; rescales init
    return model
```

### 5.4 Optimizer (`llamlam/utils.py`)

```python
from mup import MuAdamW
def build_optimizer(model, config):
    if config.parametrization == "mup":
        return MuAdamW(get_grouped_params(model, config.weight_decay, config.no_decay),
                       lr=config.learning_rate, weight_decay=config.weight_decay)
    ...  # existing AdamW / Muon / GrokAdamW
```

`MuAdamW` reads each param's `infshape.width_mult()` and divides the LR for
matrix-like params automatically — that *is* the `η/m` row of the table.

---

## 6. Path B — faithful from-scratch (no dependency)

If you want to keep llamlam's hand-rolled style, replicate the three pieces
`mup` provides: per-param width multipliers, init rescale, readout multiplier,
and LR scaling. Minimal core:

```python
# llamlam/mup.py
from dataclasses import replace
import torch, torch.nn as nn

def attach_width_mults(model, base_dims):
    """Record, per Linear/Embedding, the infinite fan-in width ratio.

    base_dims maps a dimension size -> its base size for every *infinite* dim.
    A simpler, explicit alternative to base/delta models: tag each module at
    build time with .fan_in_mult / .fan_out_mult.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fin, fout = module.in_features, module.out_features
            module.fan_in_mult  = base_dims.get(fin, 1.0)   # =fin/base if infinite else 1
            module.fan_out_mult = base_dims.get(fout, 1.0)
        elif isinstance(module, nn.Embedding):
            module.fan_in_mult = 1.0                          # vocab is finite (Input)
            module.fan_out_mult = base_dims.get(module.embedding_dim, 1.0)

def mup_rescale_and_init(model, std, readout_modules, zero_query_slices=()):
    for module in model.modules():
        if isinstance(module, nn.Embedding):                  # Input: Θ(1) init
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, nn.Linear):
            m = getattr(module, "fan_in_mult", 1.0)
            nn.init.normal_(module.weight, std=std / m**0.5)  # Hidden/Output: Θ(1/fan_in)
            if module.bias is not None: nn.init.zeros_(module.bias)
    for ro in readout_modules:                                # optional zero readout
        nn.init.zeros_(ro.weight)
    for w, sl in zero_query_slices:                           # optional zero query
        with torch.no_grad(): w[sl] = 0.0

class MuReadout(nn.Linear):
    """Linear whose output is scaled by output_mult / fan_in_width_mult."""
    def __init__(self, *a, output_mult=1.0, **k):
        super().__init__(*a, **k); self.output_mult = output_mult
    def forward(self, x):
        return super().forward(x) * (self.output_mult / getattr(self, "fan_in_mult", 1.0))

def mup_param_groups(model, weight_decay, no_decay, base_lr):
    """η for vector/Input params; η/fan_in_mult for matrix params."""
    groups = []
    name2mod = {n: m for m, ns in [(mod, [pn for pn,_ in mod.named_parameters(recurse=False)])
               for mod in model.modules()] for n in ns}  # or track during build
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
        is_matrix = p.ndim >= 2 and "embed" not in n and "pos_embed" not in n
        fan_in_mult = _lookup_fan_in_mult(model, n)          # 1.0 for vectors/embeddings
        lr = base_lr / fan_in_mult if is_matrix else base_lr
        groups.append({"params": [p], "weight_decay": wd, "lr": lr})
    return groups
```

Wiring mirrors Path A: build the model, `attach_width_mults(model, base_dims)`
where `base_dims = {dim_embd: dim_embd/base_dim_embd, 4*dim_embd: (4*dim_embd)/(4*base_dim_embd), ...}`,
call `mup_rescale_and_init`, swap the head for `MuReadout`, set the `1/dim_head`
attention scale, and build the optimizer from `mup_param_groups`. The subtlety
the lightweight recipe misses — and this captures — is that the **readout** and
the **MLP down-projection** use their *own* `fan_in_mult` (`d` and `4d`
respectively), not a single global `width_mult`.

> AdamW weight-decay note: μP's `η/m` LR also shrinks the decoupled WD step
> (`p *= 1 - lr·wd`). If you want width-stable *regularization*, scale `wd` by
> `m` for matrix params, or keep WD only on vector/Input params. Coordinate-check
> with WD on.

---

## 7. Verification — the coordinate check (this is the real test)

μP is "correct" iff per-layer activation magnitudes stay **flat as width grows**
(SP fans out). This is non-negotiable: ship μP only with a green coord check.

Using the library:

```python
from mup.coord_check import get_coord_data, plot_coord_data

def lazy(width):
    def f():
        cfg = make_cfg(dim_embd=width, dim_head=width//cfg_heads, parametrization="mup")
        return build_model(cfg)            # with set_base_shapes applied
    return f

models = {w: lazy(w) for w in (128, 256, 512, 1024, 2048)}
df = get_coord_data(models, data_loader, optimizer="adamw", lr=1e-2, nsteps=4, nseeds=3)
plot_coord_data(df, save_to=".context/extensions/mup_coordcheck.png")
```

Hand-rolled assertion for CI (`tests/test_mup_coord_check.py`, network-free):

```python
import torch
from llamlam.train import build_model
from llamlam.config import Config

def _coords(width, parametrization, steps=4):
    cfg = Config(vocab_size=64, n_layers=2, n_heads=4, dim_head=width//4,
                 max_seq_length=32, dropout=0.0, parametrization=parametrization,
                 mup_base_dim_embd=64, mup_base_dim_head=16, tie_word_embeddings=False)
    torch.manual_seed(0); model = build_model(cfg)
    ids = torch.randint(0, 64, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(); model(ids, labels=ids)["loss"].backward(); opt.step()
    out = model(ids, output_hidden_states=True)
    return [h.abs().mean().item() for h in out["hidden_states"]]

def test_mup_is_width_invariant():
    narrow = _coords(128, "mup"); wide = _coords(1024, "mup")   # 8x wider
    for a, b in zip(narrow, wide):
        assert b < 2.5 * a, f"μP coords grew with width: {a:.3f} -> {b:.3f}"

def test_sp_is_not_width_invariant():           # guards against a no-op μP
    narrow = _coords(128, "sp"); wide = _coords(1024, "sp")
    assert max(w/ (n+1e-9) for n, w in zip(narrow, wide)) > 3.0
```

Keep the coord-check PNG (μP flat vs SP fanning out) in the PR description — it
is the standard, expected artifact and the only convincing proof.

---

## 8. μTransfer workflow (the payoff)

1. Pick base widths (`mup_base_dim_embd`, `mup_base_dim_head`); confirm coord check.
2. Sweep `learning_rate` (and `input_mult`/`output_mult`) on the **small** model.
3. Lock the winners; scale `dim_embd`/`dim_head` up; **reuse** the HPs.
4. Re-run a quick coord check at the target width before the long run.

---

## 9. DiffTransformer port

After GPTModel's coord check is green, mirror it:

- `1/dim_head` score scaling in `MultiHeadDifferentialAttention` (already `1/√d`).
- Apply the Hidden μP rule to `W_q/W_k/W_v` (note: out-dim is `2·dim_head·n_heads`)
  and `W_o`; treat `head` as Output (`MuReadout`).
- λ-reparam vectors (`lambda_q1…`) and `rms_scale` are **Vector** params: Θ(1)
  init, base LR, no width scaling.
- The differential output term multiplies by `(1 - lambda_init)` and per-head
  RMSNorm — verify this composite output scale stays width-independent in the
  coord check (it should, since RMSNorm normalizes, but check explicitly).
- The `lambda_init = 0.8 - 0.6·exp(-0.3·(l-1))` depth schedule is orthogonal to
  width μP; leave it. If you also μP-scale *depth*, consult Tensor Programs VI.

---

## 10. Gotchas checklist

- [ ] **No weight tying** under μP (Input vs Output classes differ).
- [ ] Drop the SP residual `1/√(2·n_layers)` init rescale — per-layer LR covers it.
- [ ] Readout multiplier uses the readout's **own** `fan_in` ratio, MLP-down uses `4d`.
- [ ] Biases & norm gains are **Vector/Input**: Θ(1) init, base LR — never `/m`.
- [ ] Positional embedding is an **Input** weight (Θ(1) init, base LR), like tokens.
- [ ] Zero-init readout (and optionally query) for clean early dynamics.
- [ ] Save/restore **base shapes** with checkpoints (not in `state_dict`).
- [ ] Re-derive the table if you switch off Adam (SGD has different exponents).
- [ ] Coordinate-check **with** weight decay and the real optimizer, ≥3 seeds.

---

## 11. References

- Yang et al., *Tensor Programs V: Tuning Large NNs via Zero-Shot HP Transfer*,
  2022 — https://arxiv.org/abs/2203.03466
- `mup` library (MuReadout, set_base_shapes, MuAdam, coord_check) —
  https://github.com/microsoft/mup
- Yang et al., *Tensor Programs VI* (depth μP / μP+) —
  https://arxiv.org/abs/2310.02244
- Cerebras, *A Practitioner's Guide to μP*, 2023.
- min-max-gpt (the lightweight recipe this repo started from) —
  https://github.com/cloneofsimo/min-max-gpt
