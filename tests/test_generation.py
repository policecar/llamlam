"""
KV cache and generation (llamlam.generation) for GPTModel and DiffTransformer.

Tolerance for cached-vs-full logits: atol = rtol = 1e-5. Cached decoding does
the same float32 math in a different summation order (different matmul
shapes, SDPA with an explicit mask instead of is_causal), measured at <= 2e-6
for |logits| <= 4; a wrong position offset or a leaky / over-tight mask moves
logits by >= 1e-2. Prefill without a past cache runs the identical code path,
so it is compared bit-for-bit.
"""

import pytest
import torch

from llamlam.config import Config
from llamlam.difftransformer import DiffTransformer
from llamlam.generation import filter_logits, generate
from llamlam.model import GPTModel

MODELS = [GPTModel, DiffTransformer]
MAX_LEN = 16
ATOL = RTOL = 1e-5


def build(model_cls, seed=0, **overrides):
    kwargs = dict(
        max_seq_length=MAX_LEN,
        vocab_size=50,
        n_layers=2,
        n_heads=4,
        dim_head=8,
        dropout=0.0,
        bias=False,
    )
    kwargs.update(overrides)
    torch.manual_seed(seed)
    model = model_cls(Config(**kwargs))
    if model_cls is GPTModel:
        # pos_embed is initialised small (std init_std); enlarge it so positions
        # matter and a wrong cache offset changes the logits.
        with torch.no_grad():
            model.pos_embed.normal_(0, 1)
    model.eval()
    return model


def random_ids(model, batch=2, length=MAX_LEN, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, model.config.vocab_size, (batch, length), generator=g)


class IdTokenizer:
    """Space-separated token ids <-> text, so decoded output is the id list."""

    def encode(self, text, return_tensors=None):
        assert return_tensors == "pt"
        return torch.tensor([[int(t) for t in text.split()]], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens
        return " ".join(str(i) for i in ids.tolist())


def prompt_of(ids):
    return " ".join(str(i) for i in ids.tolist())


def reference_decode(model, prompt_ids, n_new, sample=None):
    """The pre-cache loop: full forward of the cropped window every step."""
    ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits = model(ids[:, -model.config.max_seq_length :])["logits"][:, -1]
            if sample is None:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                temperature, top_k, top_p = sample
                probs = torch.softmax(
                    filter_logits(logits / temperature, top_k=top_k, top_p=top_p), -1
                )
                nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, nxt), dim=1)
    return ids


# --------------------------------------------------------------------------- #
# KV cache: cached decoding reproduces the full forward
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_cls", MODELS)
def test_prefill_with_cache_is_bit_identical_to_full_forward(model_cls):
    model = build(model_cls)
    ids = random_ids(model)
    with torch.no_grad():
        full = model(ids)["logits"]
        prefill = model(ids, use_cache=True)["logits"]
    assert torch.equal(prefill, full)


@pytest.mark.parametrize("model_cls", MODELS)
def test_one_token_steps_match_full_forward(model_cls):
    model = build(model_cls)
    ids = random_ids(model)
    with torch.no_grad():
        full = model(ids)["logits"]
        out = model(ids[:, :1], use_cache=True)
        steps = [out["logits"]]
        for t in range(1, ids.size(1)):
            out = model(
                ids[:, t : t + 1],
                past_key_values=out["past_key_values"],
                use_cache=True,
            )
            steps.append(out["logits"])
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("model_cls", MODELS)
def test_multi_token_incremental_steps_match_full_forward(model_cls):
    """Several new tokens per step must attend causally among themselves."""
    model = build(model_cls)
    ids = random_ids(model)
    bounds = [0, 5, 6, 13, MAX_LEN]  # chunks of 5, 1, 7 and 3 new tokens
    with torch.no_grad():
        full = model(ids)["logits"]
        past, chunks = None, []
        for lo, hi in zip(bounds, bounds[1:]):
            out = model(ids[:, lo:hi], past_key_values=past, use_cache=True)
            past = out["past_key_values"]
            chunks.append(out["logits"])
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("model_cls", MODELS)
def test_cache_holds_one_key_value_pair_per_layer(model_cls):
    model = build(model_cls)
    ids = random_ids(model, length=7)
    with torch.no_grad():
        past = model(ids, use_cache=True)["past_key_values"]
        past = model(ids[:, :2], past_key_values=past, use_cache=True)[
            "past_key_values"
        ]
    assert len(past) == model.config.n_layers
    for k, v in past:
        assert k.shape[0] == 2 and k.shape[2] == 9
        assert v.shape[:3] == k.shape[:3]


@pytest.mark.parametrize("model_cls", MODELS)
def test_cached_mode_returns_no_loss(model_cls):
    model = build(model_cls)
    ids = random_ids(model, length=4)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        assert "loss" not in out and "past_key_values" in out
        # Reading a cache without extending it: still no loss, no new cache
        out = model(ids[:, :1], past_key_values=out["past_key_values"])
        assert "loss" not in out and "past_key_values" not in out
        assert "loss" in model(ids)


@pytest.mark.parametrize("model_cls", MODELS)
def test_cache_plus_new_tokens_beyond_max_seq_length_raises(model_cls):
    model = build(model_cls)
    ids = random_ids(model)
    with torch.no_grad():
        past = model(ids[:, : MAX_LEN - 2], use_cache=True)["past_key_values"]
        # Exactly filling the window is fine
        model(ids[:, :2], past_key_values=past, use_cache=True)
        with pytest.raises(ValueError, match="exceeds max_seq_length"):
            model(ids[:, :3], past_key_values=past, use_cache=True)
        full = model(ids, use_cache=True)["past_key_values"]
        with pytest.raises(ValueError, match="exceeds max_seq_length"):
            model(ids[:, :1], past_key_values=full, use_cache=True)
        longer = torch.cat((ids, ids[:, :1]), dim=1)
        with pytest.raises(ValueError, match="exceeds max_seq_length"):
            model(longer)


@pytest.mark.parametrize("model_cls", MODELS)
def test_invalid_cache_inputs_raise(model_cls):
    model = build(model_cls)
    ids = random_ids(model, length=4)
    with torch.no_grad():
        past = model(ids, use_cache=True)["past_key_values"]
        with pytest.raises(ValueError, match="layers"):
            model(ids[:, :1], past_key_values=past[:1], use_cache=True)
        with pytest.raises(ValueError, match="attention_mask"):
            model(ids, attention_mask=torch.ones_like(ids), use_cache=True)


# --------------------------------------------------------------------------- #
# Greedy generation with the cache == the old no-cache loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize(
    "prompt_len, n_new",
    [
        (1, 5),  # stays inside the window
        (5, MAX_LEN - 5),  # ends exactly at the window
        (5, 2 * MAX_LEN),  # crosses the window and keeps going
        (MAX_LEN - 1, 4),  # crosses right after the prompt
        (MAX_LEN, 4),  # prompt fills the window
        (MAX_LEN + 3, 4),  # prompt longer than the window is cropped
    ],
)
def test_greedy_generate_equals_no_cache_loop(model_cls, prompt_len, n_new):
    model = build(model_cls)
    prompt_ids = random_ids(model, batch=1, length=prompt_len + 1)[:, :prompt_len]
    expected = reference_decode(model, prompt_ids, n_new)
    text = model.generate(IdTokenizer(), prompt_of(prompt_ids[0]), max_new_tokens=n_new)
    assert text == prompt_of(expected[0])


@pytest.mark.parametrize("model_cls", MODELS)
def test_generate_uses_one_token_steps_until_the_window_is_full(model_cls):
    model = build(model_cls)
    lengths = []
    model.register_forward_pre_hook(
        lambda mod, args, kwargs: lengths.append(args[0].size(1)), with_kwargs=True
    )
    model.generate(IdTokenizer(), "1 2 3 4 5", max_new_tokens=15)
    # prefill 5; one-token steps up to 15 cached tokens (window is 16); after
    # that, every step re-encodes the full window
    assert lengths == [5] + [1] * 11 + [MAX_LEN] * 3


@pytest.mark.parametrize("model_cls", MODELS)
def test_generate_zero_new_tokens_returns_prompt(model_cls):
    model = build(model_cls)
    assert model.generate(IdTokenizer(), "4 5 6", max_new_tokens=0) == "4 5 6"


@pytest.mark.parametrize("model_cls", MODELS)
def test_generate_leaves_model_in_eval_mode(model_cls):
    model = build(model_cls)
    model.train()
    model.generate(IdTokenizer(), "1 2", max_new_tokens=2)
    assert not model.training


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize(
    "temperature, top_k, top_p", [(1.0, None, None), (0.7, 10, 0.9), (1.3, None, 0.8)]
)
def test_sampling_matches_no_cache_reference_under_seed(
    model_cls, temperature, top_k, top_p
):
    """Same seed -> same draws as the no-cache loop; crosses the window."""
    model = build(model_cls)
    prompt_ids = random_ids(model, batch=1, length=4)
    n_new = MAX_LEN + 4
    torch.manual_seed(123)
    expected = reference_decode(
        model, prompt_ids, n_new, sample=(temperature, top_k, top_p)
    )
    kwargs = dict(do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p)
    torch.manual_seed(123)
    a = model.generate(IdTokenizer(), prompt_of(prompt_ids[0]), n_new, **kwargs)
    torch.manual_seed(123)
    b = model.generate(IdTokenizer(), prompt_of(prompt_ids[0]), n_new, **kwargs)
    assert a == b == prompt_of(expected[0])
    greedy = model.generate(IdTokenizer(), prompt_of(prompt_ids[0]), n_new)
    if temperature == 1.0:
        # untruncated sampling on a random-init model: not the greedy path
        assert a != greedy


@pytest.mark.parametrize("model_cls", MODELS)
def test_top_k_one_equals_greedy(model_cls):
    model = build(model_cls)
    greedy = model.generate(IdTokenizer(), "1 2 3", max_new_tokens=MAX_LEN + 4)
    torch.manual_seed(0)
    sampled = model.generate(
        IdTokenizer(),
        "1 2 3",
        max_new_tokens=MAX_LEN + 4,
        do_sample=True,
        temperature=2.0,
        top_k=1,
    )
    assert sampled == greedy


def _kept(logits):
    return [
        set(torch.nonzero(row > float("-inf")).flatten().tolist()) for row in logits
    ]


# probabilities 0.5, 0.05, 0.3, 0.15 at ids 0..3; row 1 is a permutation
PROBS = torch.tensor([[0.5, 0.05, 0.3, 0.15], [0.15, 0.3, 0.05, 0.5]])
LOGITS = PROBS.log()


@pytest.mark.parametrize(
    "top_k, top_p, row0, row1",
    [
        (None, None, {0, 1, 2, 3}, {0, 1, 2, 3}),
        (1, None, {0}, {3}),
        (2, None, {0, 2}, {1, 3}),
        (3, None, {0, 2, 3}, {0, 1, 3}),
        (99, None, {0, 1, 2, 3}, {0, 1, 2, 3}),  # k capped at the vocab size
        (None, 0.4, {0}, {3}),  # top token alone reaches 0.4
        (None, 0.75, {0, 2}, {1, 3}),  # 0.5 < 0.75 <= 0.8
        (None, 0.85, {0, 2, 3}, {0, 1, 3}),  # 0.8 < 0.85 <= 0.95
        (None, 0.97, {0, 1, 2, 3}, {0, 1, 2, 3}),
        (None, 1.0, {0, 1, 2, 3}, {0, 1, 2, 3}),
        # top_p applies to the top-k renormalised distribution: after top_k=3
        # the masses are 0.526, 0.316, 0.158 -> cumulative 0.842 before the third
        (3, 0.8, {0, 2}, {1, 3}),
        (3, 0.9, {0, 2, 3}, {0, 1, 3}),
    ],
)
def test_filter_logits_keeps_expected_tokens(top_k, top_p, row0, row1):
    filtered = filter_logits(LOGITS, top_k=top_k, top_p=top_p)
    assert _kept(filtered) == [row0, row1]
    kept = filtered > float("-inf")
    assert torch.equal(filtered[kept], LOGITS[kept])  # kept logits unchanged


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(do_sample=True, temperature=0.0),
        dict(do_sample=True, temperature=-1.0),
        dict(do_sample=True, temperature=float("nan")),
        dict(do_sample=True, top_k=0),
        dict(do_sample=True, top_k=2.5),
        dict(do_sample=True, top_k=True),
        dict(do_sample=True, top_p=0.0),
        dict(do_sample=True, top_p=-0.1),
        dict(do_sample=True, top_p=1.1),
        dict(top_k=5),  # sampling arguments without do_sample
        dict(top_p=0.9),
        dict(temperature=0.5),
        dict(max_new_tokens=-1),
    ],
)
def test_invalid_generation_arguments_raise(kwargs):
    model = build(GPTModel)
    kwargs = {"max_new_tokens": 3, **kwargs}
    with pytest.raises(ValueError):
        generate(model, IdTokenizer(), "1 2", **kwargs)


def test_empty_prompt_raises():
    model = build(GPTModel)
    with pytest.raises(ValueError, match="zero tokens"):
        model.generate(IdTokenizer(), "", max_new_tokens=3)
