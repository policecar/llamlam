"""
Regression tests for review fixes shared by GPTModel and DiffTransformer:
padding-aware loss, bias flag, str output_dir, causality, generate device,
memorisation.
"""

import string

import pytest
import torch
import torch.nn.functional as F

from llamlam.config import Config
from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel

MODELS = [GPTModel, DiffTransformer]


def small_config(**overrides):
    kwargs = dict(
        max_seq_length=16,
        vocab_size=50,
        n_layers=2,
        n_heads=2,
        dim_head=16,
        dropout=0.0,
        bias=False,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def build(model_cls, seed=0, **overrides):
    torch.manual_seed(seed)
    model = model_cls(small_config(**overrides))
    model.eval()
    return model


def random_ids(config, batch=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, config.vocab_size, (batch, config.max_seq_length), generator=g
    )


# --------------------------------------------------------------------------- #
# Item 1: padding excluded from the loss
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_cls", MODELS)
def test_masked_loss_matches_manual_loss_over_real_targets(model_cls):
    model = build(model_cls)
    input_ids = random_ids(model.config)
    seq = input_ids.size(1)
    # Right padding of different lengths per row
    lengths = [seq, 10, 5]
    attention_mask = torch.zeros_like(input_ids)
    for row, n in enumerate(lengths):
        attention_mask[row, :n] = 1

    with torch.no_grad():
        out_masked = model(input_ids, attention_mask=attention_mask)
        out_plain = model(input_ids)

    # Logits are unaffected by the mask
    assert torch.equal(out_masked["logits"], out_plain["logits"])

    # Manual loss: mean CE over targets t whose mask is 1
    logits = out_plain["logits"][:, :-1, :]
    targets = input_ids[:, 1:]
    keep = attention_mask[:, 1:].bool()
    expected = F.cross_entropy(logits[keep], targets[keep])
    n_kept = int(keep.sum())
    assert n_kept == sum(n - 1 for n in lengths)

    torch.testing.assert_close(out_masked["loss"], expected)
    # And it really differs from the padding-polluted loss
    assert not torch.allclose(out_masked["loss"], out_plain["loss"])


@pytest.mark.parametrize("model_cls", MODELS)
def test_all_ones_mask_and_no_mask_give_identical_loss(model_cls):
    model = build(model_cls)
    input_ids = random_ids(model.config)
    with torch.no_grad():
        plain = model(input_ids)["loss"]
        ones = model(input_ids, attention_mask=torch.ones_like(input_ids))["loss"]
        float_ones = model(
            input_ids, attention_mask=torch.ones(input_ids.shape, dtype=torch.float)
        )["loss"]
    assert torch.equal(plain, ones)
    assert torch.equal(plain, float_ones)


@pytest.mark.parametrize("model_cls", MODELS)
def test_fully_masked_targets_raise(model_cls):
    model = build(model_cls)
    input_ids = random_ids(model.config)
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[:, 0] = 1  # only the first token is real -> no real target
    with pytest.raises(ValueError, match="masks every target"):
        model(input_ids, attention_mask=attention_mask)


@pytest.mark.parametrize("model_cls", MODELS)
def test_mask_shape_mismatch_raises(model_cls):
    model = build(model_cls)
    input_ids = random_ids(model.config)
    with pytest.raises(ValueError, match="shape"):
        model(input_ids, attention_mask=torch.ones(input_ids.size(0), 3))


@pytest.mark.parametrize("model_cls", MODELS)
def test_padded_target_tokens_do_not_affect_masked_loss(model_cls):
    """Changing a padded target token must not change the masked loss."""
    model = build(model_cls)
    input_ids = random_ids(model.config)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -1] = 0
    altered = input_ids.clone()
    altered[:, -1] = (altered[:, -1] + 1) % model.config.vocab_size
    with torch.no_grad():
        a = model(input_ids, attention_mask=attention_mask)["loss"]
        b = model(altered, attention_mask=attention_mask)["loss"]
    # Last position is only ever a target (never a source for earlier logits)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Item 6: GPTModel honours config.bias
# --------------------------------------------------------------------------- #


def _bias_param_names(model):
    return [
        n
        for n, _ in model.named_parameters()
        if n.endswith(".bias") and (n.startswith("blocks.") or n.startswith("ln_f."))
    ]


def test_gpt_bias_false_has_no_bias_parameters():
    model = build(GPTModel, bias=False)
    assert _bias_param_names(model) == []
    assert model.ln_f.bias is None
    for block in model.blocks:
        for layer in block.feedforward:
            if isinstance(layer, torch.nn.Linear):
                assert layer.bias is None


def test_gpt_bias_true_has_bias_parameters():
    model = build(GPTModel, bias=True)
    names = set(_bias_param_names(model))
    assert "ln_f.bias" in names
    for i in range(model.config.n_layers):
        assert f"blocks.{i}.feedforward.0.bias" in names
        assert f"blocks.{i}.feedforward.2.bias" in names
        assert f"blocks.{i}.norm_1.bias" in names
        assert f"blocks.{i}.norm_2.bias" in names
    # Feed-forward biases are zero-initialised
    assert torch.count_nonzero(model.blocks[0].feedforward[0].bias) == 0


# --------------------------------------------------------------------------- #
# Item 7: save_pretrained accepts str and Path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("as_str", [True, False])
def test_save_pretrained_accepts_str_and_path(model_cls, as_str, tmp_path):
    model = build(model_cls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out_dir = tmp_path / "nested" / "ckpt"
    model.save_pretrained(str(out_dir) if as_str else out_dir, "t1", optimizer)

    model_file = out_dir / "model_t1.pt"
    assert model_file.is_file()
    assert (out_dir / "optimizer_t1.pt").is_file()

    reloaded = model_cls.from_pretrained(
        model.config, {"model_name_or_path": str(model_file)}
    )
    for (n1, p1), (n2, p2) in zip(
        model.state_dict().items(), reloaded.state_dict().items()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)


# --------------------------------------------------------------------------- #
# Causality and memorisation (both models)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_cls", MODELS)
def test_causality(model_cls):
    model = build(model_cls)
    input_ids = random_ids(model.config, batch=2)
    t = 7
    altered = input_ids.clone()
    altered[:, t] = (altered[:, t] + 1) % model.config.vocab_size
    with torch.no_grad():
        a = model(input_ids)["logits"]
        b = model(altered)["logits"]
    torch.testing.assert_close(a[:, :t], b[:, :t], rtol=0, atol=1e-5)
    assert not torch.allclose(a[:, t], b[:, t])


# --------------------------------------------------------------------------- #
# generate() runs on the model's device and never moves the model
# --------------------------------------------------------------------------- #


class StubTokenizer:
    """Character-level tokenizer over the first vocab_size ASCII letters."""

    def __init__(self, vocab_size):
        self.alphabet = (string.ascii_letters + string.digits)[:vocab_size]
        assert len(self.alphabet) == vocab_size

    def encode(self, text, return_tensors=None):
        assert return_tensors == "pt"
        return torch.tensor([[self.alphabet.index(c) for c in text]])

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens
        return "".join(self.alphabet[i] for i in ids.tolist())


def _greedy_reference(model, ids, n_new):
    """Greedy decoding written out, cropping the context like generate()."""
    with torch.no_grad():
        for _ in range(n_new):
            logits = model(ids[:, -model.config.max_seq_length :])["logits"]
            ids = torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    return ids


@pytest.mark.parametrize("model_cls", MODELS)
def test_generate_keeps_cpu_model_on_cpu(model_cls, monkeypatch):
    """Even when an accelerator is reported available, a CPU model stays put."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = build(model_cls)
    tokenizer = StubTokenizer(model.config.vocab_size)
    prompt = "abcde"
    n_new = 20  # > max_seq_length - len(prompt): exercises context cropping

    text = model.generate(tokenizer, prompt, max_new_tokens=n_new)

    assert {p.device.type for p in model.parameters()} == {"cpu"}
    assert {b.device.type for b in model.buffers()} <= {"cpu"}
    assert len(text) == len(prompt) + n_new
    assert text.startswith(prompt)
    expected = _greedy_reference(
        model, tokenizer.encode(prompt, return_tensors="pt"), n_new
    )
    assert text == tokenizer.decode(expected[0], skip_special_tokens=True)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("model_cls", MODELS)
def test_generate_runs_on_model_device_mps(model_cls):
    model = build(model_cls).to("mps")
    tokenizer = StubTokenizer(model.config.vocab_size)
    text = model.generate(tokenizer, "abc", max_new_tokens=5)
    assert {p.device.type for p in model.parameters()} == {"mps"}
    assert len(text) == 3 + 5 and text.startswith("abc")


@pytest.mark.parametrize("model_cls", MODELS)
def test_memorises_small_random_batch(model_cls):
    model = build(model_cls)
    model.train()
    input_ids = random_ids(model.config, batch=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial = model(input_ids)["loss"].item()
    for _ in range(300):
        optimizer.zero_grad()
        loss = model(input_ids)["loss"]
        loss.backward()
        optimizer.step()
    final = loss.item()
    # ln(50) ~ 3.9 at chance; memorised means near zero
    assert initial > 3.0
    assert final < 0.1, f"{model_cls.__name__} final loss {final:.3f}"
