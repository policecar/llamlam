"""
Standard parametrisation (llamlam/model.py DEC-002) and weight tying for both
models: the 1/sqrt(dim_head) attention scale on the full and the cached path,
tied input embedding / output head (same tensor, parameter counts, gradients
from both uses, save/load with torch and through Accelerate), and an initial
loss near ln(vocab_size) at the default config.
"""

import math

import pytest
import torch
from accelerate import Accelerator

from llamlam.config import Config
from llamlam.difftransformer import DiffTransformer
from llamlam.model import Context, GPTModel

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


def embedding_of(model):
    return model.embed.weight if isinstance(model, GPTModel) else model.token_emb.weight


def random_ids(config, batch=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, config.vocab_size, (batch, config.max_seq_length), generator=g
    )


##########################################
# Attention scale
##########################################


def manual_causal_attention(context, x):
    """softmax(q k^T / sqrt(d) + causal mask) v, then out_proj, written out."""
    batch, length, dim = x.shape
    h, d = context.n_heads, context.dim_head
    q, k, v = context.qkv_proj(x).view(batch, length, h, 3 * d).chunk(3, dim=-1)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, h, L, d]
    scores = q @ k.transpose(-1, -2) / math.sqrt(d)
    future = torch.ones(length, length, dtype=torch.bool).triu(diagonal=1)
    weights = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    out = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
    return context.out_proj(out)


def test_attention_scale_is_one_over_sqrt_dim_head():
    context = Context(dim_embd=64, n_heads=4)
    assert context.dim_head == 16
    assert context.scaling == pytest.approx(1 / 4)


def test_attention_matches_manual_standard_scaled_softmax():
    torch.manual_seed(0)
    context = Context(dim_embd=64, n_heads=4)
    with torch.no_grad():
        context.qkv_proj.weight.normal_(0, 0.5)  # logits large enough to matter
    x = torch.randn(2, 10, 64)
    with torch.no_grad():
        expected = manual_causal_attention(context, x)
        full = context(x)
        # cached path: prefill 6 positions, then 4 more against the cache
        head, present = context(x[:, :6], use_cache=True)
        tail = context(x[:, 6:], past_kv=present)
    torch.testing.assert_close(full, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        torch.cat((head, tail), dim=1), expected, rtol=1e-5, atol=1e-5
    )


##########################################
# Weight tying
##########################################


@pytest.mark.parametrize("model_cls", MODELS)
def test_tie_word_embeddings_is_on_by_default(model_cls):
    assert Config().tie_word_embeddings is True
    model = model_cls(small_config())
    assert model.head.weight is embedding_of(model)
    model = model_cls(small_config(tie_word_embeddings=False))
    assert model.head.weight is not embedding_of(model)
    assert model.head.weight.data_ptr() != embedding_of(model).data_ptr()


@pytest.mark.parametrize("model_cls", MODELS)
def test_tying_removes_exactly_one_vocab_by_dim_matrix(model_cls):
    config = Config()  # default size
    with torch.device("meta"):
        tied = model_cls(config)
        untied = model_cls(Config(tie_word_embeddings=False))
    n_tied = sum(p.numel() for p in tied.parameters())
    n_untied = sum(p.numel() for p in untied.parameters())
    assert n_untied - n_tied == config.vocab_size * config.dim_embd
    assert len(list(untied.parameters())) - len(list(tied.parameters())) == 1


def test_parameter_counts_at_default_config():
    """The totals quoted in the README / review report, tied and untied."""
    expected = {
        (GPTModel, True): 124_337_664,
        (GPTModel, False): 162_935_040,
        (DiffTransformer, True): 124_400_640,
        (DiffTransformer, False): 162_998_016,
    }
    for (model_cls, tie), count in expected.items():
        with torch.device("meta"):
            model = model_cls(Config(tie_word_embeddings=tie))
        assert sum(p.numel() for p in model.parameters()) == count, (model_cls, tie)


@pytest.mark.parametrize("model_cls", MODELS)
def test_tied_gradient_is_sum_of_embedding_and_head_gradients(model_cls):
    """The shared matrix gets the gradient of both uses (input and output)."""
    torch.manual_seed(0)
    tied = model_cls(small_config())
    untied = model_cls(small_config(tie_word_embeddings=False))
    # same values everywhere, the untied head a separate copy of the embedding
    untied.load_state_dict(tied.state_dict())
    assert torch.equal(untied.head.weight, embedding_of(untied))
    input_ids = random_ids(tied.config)

    loss_tied = tied(input_ids)["loss"]
    loss_untied = untied(input_ids)["loss"]
    assert torch.equal(loss_tied, loss_untied)
    loss_tied.backward()
    loss_untied.backward()

    g_input = embedding_of(untied).grad
    g_output = untied.head.weight.grad
    assert g_input.abs().sum() > 0 and g_output.abs().sum() > 0  # both uses matter
    torch.testing.assert_close(embedding_of(tied).grad, g_input + g_output)


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("tie", [True, False])
def test_save_and_from_pretrained_keep_tying(model_cls, tie, tmp_path):
    torch.manual_seed(0)
    model = model_cls(small_config(tie_word_embeddings=tie))
    with torch.no_grad():
        embedding_of(model).add_(1.0)  # differs from any fresh init
    model.save_pretrained(tmp_path, "t")
    torch.manual_seed(1)
    reloaded = model_cls.from_pretrained(
        model.config, {"model_name_or_path": str(tmp_path / "model_t.pt")}
    )
    assert (reloaded.head.weight is embedding_of(reloaded)) is tie
    assert torch.equal(embedding_of(reloaded), embedding_of(model))
    assert torch.equal(reloaded.head.weight, model.head.weight)


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("tie", [True, False])
def test_accelerate_save_and_load_state_keep_tying(model_cls, tie, tmp_path):
    """save_state writes safetensors, which stores a tied matrix once; loading
    into a freshly built model must leave it tied and equal to the saved one."""
    config = small_config(tie_word_embeddings=tie)
    accelerator = Accelerator(cpu=True)
    torch.manual_seed(0)
    model = accelerator.prepare(model_cls(config))
    with torch.no_grad():
        embedding_of(model).add_(1.0)
        if not tie:
            model.head.weight.sub_(1.0)
    saved = {n: t.clone() for n, t in model.state_dict().items()}
    accelerator.save_state(str(tmp_path))

    restorer = Accelerator(cpu=True)
    torch.manual_seed(1)
    fresh = restorer.prepare(model_cls(config))
    assert not torch.equal(embedding_of(fresh), saved_embedding(saved, model_cls))
    restorer.load_state(str(tmp_path))

    assert (fresh.head.weight is embedding_of(fresh)) is tie
    for name, tensor in fresh.state_dict().items():
        assert torch.equal(tensor, saved[name]), name
    # still tied in use: an update through the head is seen by the embedding
    if tie:
        with torch.no_grad():
            fresh.head.weight.zero_()
        assert torch.all(embedding_of(fresh) == 0)


def saved_embedding(state, model_cls):
    return state["embed.weight" if model_cls is GPTModel else "token_emb.weight"]


##########################################
# Initial loss
##########################################


@pytest.mark.parametrize("model_cls", MODELS)
def test_initial_loss_is_near_ln_vocab_at_default_config(model_cls):
    """At init the prediction is close to uniform: loss ~= ln(vocab_size).

    Random logits of std s add about s^2 / 2 nats; measured +0.16 (GPTModel,
    logit std 0.55) and +0.05 (DiffTransformer) at the default config. A
    tied head that predicted the input token, or a too-large init, would add
    far more.
    """
    config = Config()
    torch.manual_seed(0)
    model = model_cls(config).eval()
    assert model.head.weight is embedding_of(model)  # the default, tied
    g = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (2, 64), generator=g)
    with torch.no_grad():
        loss = model(input_ids)["loss"].item()
    assert abs(loss - math.log(config.vocab_size)) < 0.25, loss


##########################################
# Config defaults
##########################################


def test_init_std_default_and_effect():
    assert Config().init_std == 0.02
    torch.manual_seed(0)
    model = GPTModel(small_config(init_std=0.1, vocab_size=1000))
    assert model.embed.weight.std().item() == pytest.approx(0.1, rel=0.05)


def test_old_alpha_argument_is_gone():
    with pytest.raises(TypeError):
        GPTModel(small_config(), alpha=0.5)
    with pytest.raises(TypeError):
        Context(32, 2, alpha=0.5)


def test_difftransformer_shared_matrix_keeps_xavier_init():
    """Tying changes no init: the shared matrix is token_emb's xavier_uniform."""
    torch.manual_seed(0)
    config = small_config(vocab_size=1000, dim_head=32)
    model = DiffTransformer(config)
    bound = math.sqrt(6 / (config.vocab_size + config.dim_embd))
    w = model.token_emb.weight
    assert w.abs().max().item() <= bound
    assert w.std().item() == pytest.approx(bound / math.sqrt(3), rel=0.05)
