"""
Autoregressive generation shared by GPTModel and DiffTransformer.

Both models expose forward(input_ids, past_key_values=None, use_cache=False)
returning {"logits", "past_key_values"} and a config with max_seq_length.
"""

import torch


def validate_sampling_args(do_sample, temperature, top_k, top_p):
    """
    Raises:
        ValueError: temperature <= 0, top_k not an int >= 1, top_p outside
            (0, 1], or a sampling argument set while do_sample is False.
    """
    if not temperature > 0:  # also rejects NaN
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError(f"top_k must be an int >= 1 or None, got {top_k!r}")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1] or None, got {top_p}")
    if not do_sample and (temperature != 1.0 or top_k is not None or top_p is not None):
        raise ValueError(
            "temperature / top_k / top_p only apply with do_sample=True; "
            "greedy decoding would ignore them"
        )


def filter_logits(logits, top_k=None, top_p=None):
    """
    Top-k then nucleus (top-p) filtering of [..., vocab] logits.

    top_k keeps exactly the k highest logits (k is capped at the vocab size).
    top_p keeps the smallest set of most probable tokens whose total
    probability reaches top_p: a token is kept iff the probability mass of the
    tokens ranked above it is < top_p, so the top token is always kept and
    top_p = 1 keeps everything. Removed entries are set to -inf.
    """
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(-1, torch.topk(logits, k, dim=-1).indices, True)
        logits = logits.masked_fill(~keep, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        mass_above = sorted_probs.cumsum(dim=-1) - sorted_probs
        remove_sorted = mass_above >= top_p
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    do_sample=False,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """
    Generate text; greedy by default, do_sample=True samples from the
    temperature-scaled, top-k / top-p filtered distribution (torch's global
    RNG, so torch.manual_seed makes it reproducible).

    Runs on the model's current device and never moves it; puts the model in
    eval mode (and leaves it there).

    @decision DEC-001: Use the KV cache (one token per step) while the
    sequence fits in max_seq_length; once it no longer fits, re-encode the
    last max_seq_length tokens without a cache on every step.
    @rationale: Position embeddings are learned and absolute. Cropping the
    window shifts every kept token to a new position, so cached keys/values
    (computed at the old positions) are invalid; there is no exact sliding
    cache. Re-encoding gives exactly the old no-cache loop's output, at the old
    loop's cost (one full-window forward per token) past the window.

    Returns:
        The decoded prompt + generated tokens.
    """
    validate_sampling_args(do_sample, temperature, top_k, top_p)
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

    device = next(model.parameters()).device
    model.eval()

    token_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if token_ids.size(1) == 0:
        raise ValueError("prompt encodes to zero tokens")
    max_len = model.config.max_seq_length
    past = None

    for _ in range(max_new_tokens):
        # A cache only exists while nothing was cropped (cache length < max_len),
        # so the cached positions are the tokens' true positions 0..len-1.
        if past is not None and past[0][0].size(2) < max_len:
            out = model(token_ids[:, -1:], past_key_values=past, use_cache=True)
        else:
            out = model(token_ids[:, -max_len:], use_cache=True)
        past = out["past_key_values"]
        logits = out["logits"][:, -1, :]

        if do_sample:
            logits = filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
            idx_next = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        token_ids = torch.cat((token_ids, idx_next), dim=1)

    return tokenizer.decode(token_ids[0], skip_special_tokens=True)
