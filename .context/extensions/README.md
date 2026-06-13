# `.context/extensions/`

Design / implementation guides for features not yet in the codebase. Each is a
self-contained spec a future contributor can implement from. Nothing here
changes runtime behavior.

## μP (Maximal Update Parametrization)

Two write-ups for adding `parametrization="mup"` — pick by how you intend to
scale and how strong a guarantee you need.

| | [`mup-lightweight.md`](./mup-lightweight.md) | [`mup-full.md`](./mup-full.md) |
|---|---|---|
| Width tracking | one global `width_mult = dim_embd / base` | per-parameter `fan_in` ratios from **base shapes** |
| Correct when you scale… | `dim_embd` alone (heads/MLP-ratio fixed) | `dim_embd`, `dim_head`, `n_heads`, MLP ratio **independently** |
| Readout scaling | `1/width_mult` (approx) | true `Θ(1/fan_in)` (e.g. MLP-down uses `4d`) |
| Dependency | none (hand-rolled) | `mup` library *or* a from-scratch `InfShape` tracker |
| Effort | ~an afternoon | larger; base/delta models + LR wrapper |
| μTransfer guarantee | informal | the real thing (Tensor Programs V) |
| Verification | coordinate check (required) | coordinate check (required) |

**Recommendation:** prototype with the lightweight version to get a quick win on
a single width axis. The moment you scale more than `dim_embd` alone — or you
want trustworthy μTransfer to a much larger model — move to the full version.
Both ship only behind a **green coordinate check** (activations flat as width
grows); that test is identical in spirit across both and is the only convincing
proof the parametrization is right.

Shared facts (history, the `1/dim_head` attention that was really μP, the
commented-out LR coupling in `utils.get_grouped_params`) are repeated in each
doc so they stand alone.
