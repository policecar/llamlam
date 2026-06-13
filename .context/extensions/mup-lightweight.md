# Extension: Maximal Update Parametrization (μP) — lightweight version

Status: **design / implementation guide** (not yet implemented).
Audience: a contributor who wants a quick μP in llamlam keyed off a single
global `width_mult`.

> This is the *shortcut*. It is correct only when every infinite dimension
> scales together by the same factor and every hidden matrix is square in width.
> For independent scaling of `dim_embd` / `dim_head` / `n_heads` / the MLP ratio,
> or true μTransfer guarantees, use [`mup-full.md`](./mup-full.md). See the
> [index](./README.md) for the trade-off.

## Why μP

Under standard parametrization (SP) the optimal learning rate (and other HPs)
shifts as you change model width, so HPs tuned on a small model don't transfer
to a big one. **μP** (Yang & Hu, *Tensor Programs V*, 2022) reparametrizes init
scale, per-layer learning rates, and the attention/output multipliers so the
optimal HPs become (approximately) **width-invariant** ("μTransfer"): tune
`learning_rate` on a tiny model, scale width up, reuse it.

## What's already here (and what was missing)

`llamlam` was cribbed from
[min-max-gpt](https://github.com/cloneofsimo/min-max-gpt), itself a μP repo, and
fingerprints survived in the pre-refactor code:

- `model.py` attention used `scale = 1/dim_head` — the μP **1/d attention**
  (SP wants `1/sqrt(dim_head)`).
- Embedding init `std ≈ 1.65` — a width-independent **O(1)** init.
- Head init `std ≈ 1/(2·dim_embd)` — the μP **O(1/width)** readout.
- `utils.get_grouped_params` had commented-out per-group LR:
  `lr * (3.3 if is_embed else 1.0)` and `lr * (1/dim_head)`.

So the repo was ~80% of a (lightweight) μP. The **missing pieces** were the
per-group LR coupling (commented out) and the output-logit multiplier — without
them the init alone just destabilizes. The refactor moved the default to clean
SP; this doc adds the lightweight μP back as an option.

## Recipe (lightweight / "simple μP", Adam)

Let `width_mult = dim_embd / mup_base_dim_embd`. SP is recovered at
`width_mult == 1`. Tune at the base width; scale by changing `dim_embd` only.

| Component | Init std | Forward multiplier | Adam LR |
|---|---|---|---|
| token/pos embedding | `init_std` (constant) | `input_mult` | `base_lr` |
| hidden matrices (qkv, out_proj, mlp) | `init_std / sqrt(width_mult)` | 1 | `base_lr / width_mult` |
| output head | `init_std / sqrt(width_mult)` | `output_mult / width_mult` | `base_lr / width_mult` |
| biases, LayerNorm/RMSNorm gains | 0 / 1 | 1 | `base_lr` |
| attention scores | — | scale by **`1/dim_head`** (not `1/sqrt`) | — |

`input_mult` / `output_mult` are O(1) tunable constants (default 1.0; the old
code's `3.3` was effectively an `input_mult`). Turn **off** weight tying under
μP — embedding (O(1) init) and head (O(1/width) effective scale) differ.

## Implementation sketch (GPTModel)

Recommended scope: **GPTModel only** first; DiffTransformer note at the end. All
snippets reference current symbols (`llamlam/config.py`, `llamlam/model.py`,
`llamlam/utils.py`).

### 1. Config (`llamlam/config.py`)

```python
parametrization: str = "sp"        # "sp" (standard) or "mup"
mup_base_dim_embd: int = 256       # base width; width_mult = dim_embd / this
mup_input_mult: float = 1.0        # multiplier on embedding output
mup_output_mult: float = 1.0       # multiplier on logits (applied as / width_mult)
```

Under μP also set `tie_word_embeddings=False`.

### 2. Attention scale (`Context`)

Make the SDPA scale explicit and μP-aware (pass it from `Block`/`GPTModel`):

```python
class Context(nn.Module):
    def __init__(self, dim_embd, n_heads, attn_scale=None):
        ...
        self.attn_scale = attn_scale            # None -> SDPA default (1/sqrt d)

    def forward(self, x, attn_mask=None):
        ...
        if attn_mask is None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.attn_scale)
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=self.attn_scale)
```

with `attn_scale = (1.0 / dim_head) if parametrization == "mup" else None`.

### 3. Init + multipliers (`GPTModel`)

```python
def __init__(self, config):
    ...
    self.mup = config.parametrization == "mup"
    self.width_mult = config.dim_embd / config.mup_base_dim_embd if self.mup else 1.0
    self.output_mult = (config.mup_output_mult / self.width_mult) if self.mup else 1.0
    self.input_mult = config.mup_input_mult if self.mup else 1.0
    self.apply(self._init_weights)
    # Under μP the per-layer LR controls the residual-stream scale, so you may
    # drop the SP residual 1/sqrt(2*n_layers) rescale — coordinate-check both.

def _init_weights(self, module):
    std = self.config.init_std
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=std)                 # O(1)
    elif isinstance(module, nn.Linear):
        s = std / (self.width_mult ** 0.5) if self.mup else std
        nn.init.normal_(module.weight, std=s)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False):
    ...
    x = self.embed(input_ids) * self.input_mult + self.pos_embed[:, : input_ids.size(1), :]
    ...
    logits = (self.head(x) * self.output_mult).float()
```

### 4. Per-group learning rates (`llamlam/utils.py`)

```python
def get_grouped_params(model, weight_decay=0.1, no_decay=(), mup=False, width_mult=1.0):
    groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
        group = {"params": [p], "weight_decay": wd}
        if mup:
            is_vector = p.ndim < 2 or "embed" in n or "pos_embed" in n
            group["lr_mult"] = 1.0 if is_vector else (1.0 / width_mult)
        groups.append(group)
    return groups
```

Then in `build_optimizer`, fold `lr_mult` into the concrete LR:

```python
groups = get_grouped_params(model, config.weight_decay, config.no_decay,
                            mup=(config.parametrization == "mup"),
                            width_mult=getattr(model, "width_mult", 1.0))
for g in groups:
    g["lr"] = config.learning_rate * g.pop("lr_mult", 1.0)
return AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
```

## Verification: coordinate check

Even the shortcut must be coordinate-checked (activations flat as width grows).
Add `tests/test_mup_coord_check.py`:

```python
import torch
from llamlam.config import Config
from llamlam.model import GPTModel

def _avg_abs_acts(width, parametrization, steps=3):
    cfg = Config(vocab_size=64, n_layers=2, n_heads=4, dim_head=width // 4,
                 max_seq_length=32, dropout=0.0, parametrization=parametrization,
                 mup_base_dim_embd=64, tie_word_embeddings=False)
    torch.manual_seed(0); model = GPTModel(cfg)
    ids = torch.randint(0, 64, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(); model(ids, labels=ids)["loss"].backward(); opt.step()
    out = model(ids, output_hidden_states=True)
    return [h.abs().mean().item() for h in out["hidden_states"]]

def test_mup_activations_are_width_stable():
    a = _avg_abs_acts(128, "mup"); b = _avg_abs_acts(512, "mup")  # 4x wider
    for x, y in zip(a, b):
        assert y < 3 * x, f"μP activations grew with width: {x:.3f} -> {y:.3f}"
```

Plot `_avg_abs_acts` across widths for both `"sp"` and `"mup"`: SP fans out,
μP stays flat. Keep the plot in the PR. **If you scale more than `dim_embd`
alone and the coord check fans out, that's the cue to switch to the full
version.**

## DiffTransformer note

Diff attention already scales `1/sqrt(dim_head)` and has extra structure (two
softmaxes, the `(1 - lambda_init)` output scale, per-head RMSNorm). A μP port
needs `1/dim_head` scaling, the init/LR table on `W_q/W_k/W_v/W_o` and `head`,
and treating RMSNorm gains + λ-reparam vectors as O(1) vector params. Do
GPTModel first, get the coord check green, then mirror it.

## References

- Yang et al., *Tensor Programs V*, 2022 — https://arxiv.org/abs/2203.03466
- `mup` library — https://github.com/microsoft/mup
- Cerebras, *A Practitioner's Guide to μP*, 2023.
- min-max-gpt — https://github.com/cloneofsimo/min-max-gpt
