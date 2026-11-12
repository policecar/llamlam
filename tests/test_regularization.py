import pytest
import torch
from torch.optim import Adam, AdamW

from llamlam.config import Config
from llamlam.model import GPTModel, LayerNorm
from llamlam.utils import get_grouped_params


def small_config(**overrides):
    kwargs = dict(max_seq_length=16, vocab_size=100, n_layers=2, n_heads=2, dim_head=8)
    kwargs.update(overrides)
    return Config(**kwargs)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def model():
    return GPTModel(small_config())  # default dropout (0.1), needed by test_dropout


@pytest.fixture
def det_model():
    """Same architecture without dropout, for deterministic optimisation tests."""
    return GPTModel(small_config(dropout=0.0))


@pytest.fixture
def train_config():
    return Config()


def test_weight_decay(det_model, train_config):
    """Weight decay shrinks exactly the decayed weights, by lr * wd * weight.

    One AdamW step with the repo's param groups from the same state, with and
    without decay: the gradients and Adam update are identical, and decoupled
    decay first scales each decayed weight by (1 - lr * wd). So every
    Linear/Embedding weight ends lr * wd * its initial value lower, and every
    other parameter is untouched by decay. (Comparing summed norms after 100
    steps instead is chaotic: at wd 0.01 its sign varied with the seed.)
    """
    input_ids = torch.randint(0, 100, (4, 16))
    init_state = det_model.state_dict()
    lr, wd = 1e-3, train_config.weight_decay
    assert wd > 0

    def one_step(weight_decay):
        model = GPTModel(det_model.config)
        model.load_state_dict(init_state)
        groups = get_grouped_params(model, weight_decay=weight_decay)
        decayed = {id(p) for p in groups[0]["params"]}
        optimizer = AdamW(groups, lr=lr)
        model(input_ids)["loss"].backward()
        optimizer.step()
        params = dict(model.named_parameters())
        return params, {n for n, p in params.items() if id(p) in decayed}

    without, _ = one_step(0.0)
    with_decay, decayed = one_step(wd)

    assert {"embed.weight", "blocks.0.context.qkv_proj.weight"} <= decayed
    assert "pos_embed" not in decayed and "ln_f.weight" not in decayed
    for name, param in with_decay.items():
        if name in decayed:
            torch.testing.assert_close(
                without[name] - param,
                lr * wd * init_state[name],
                rtol=0,
                atol=1e-8,
                msg=f"{name} not decayed by lr * wd * weight",
            )
        else:
            assert torch.equal(param, without[name]), f"{name} was decayed"
    norm_with = sum(with_decay[n].norm().item() for n in decayed)
    norm_without = sum(without[n].norm().item() for n in decayed)
    assert norm_with < norm_without, "Weight decay did not reduce parameter norms"


def test_gradient_clipping(det_model, train_config):
    max_norm = train_config.gradient_clipping
    input_ids = torch.randint(0, 100, (4, 16))

    # Scale the loss so the gradient norm is well above the clipping threshold
    (10 * det_model(input_ids)["loss"]).backward()
    initial_grads = [p.grad.clone() for p in det_model.parameters()]
    initial_grad_norm = torch.nn.utils.clip_grad_norm_(
        det_model.parameters(), float("inf")
    )
    assert initial_grad_norm > 2 * max_norm, "precondition: gradients need clipping"

    torch.nn.utils.clip_grad_norm_(det_model.parameters(), max_norm)
    clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
        det_model.parameters(), float("inf")
    )

    assert clipped_grad_norm <= max_norm + 1e-5, (
        "Gradient clipping did not limit gradient norm"
    )
    assert clipped_grad_norm == pytest.approx(max_norm, rel=1e-4)

    # Clipping rescales, it does not change the gradient direction
    scale = (max_norm / initial_grad_norm).item()
    for g0, p in zip(initial_grads, det_model.parameters()):
        torch.testing.assert_close(p.grad, g0 * scale, rtol=1e-4, atol=1e-8)


def test_dropout(model):
    assert model.config.dropout > 0
    model.eval()
    input_ids = torch.randint(0, 100, (4, 16))

    with torch.no_grad():
        output1 = model(input_ids)["logits"]
        output2 = model(input_ids)["logits"]

    assert torch.allclose(output1, output2), "Model outputs differ in eval mode"

    model.train()
    output1 = model(input_ids)["logits"]
    output2 = model(input_ids)["logits"]

    assert not torch.allclose(output1, output2), (
        "Model outputs are identical in train mode"
    )


def test_layer_norm(model):
    norms = [
        m for m in model.modules() if isinstance(m, (torch.nn.LayerNorm, LayerNorm))
    ]
    # norm_1 and norm_2 in every block (llamlam LayerNorm) plus ln_f (torch)
    assert len(norms) == 2 * model.config.n_layers + 1

    for module in norms:
        dim = module.weight.shape[0]
        x = 3 * torch.randn(4, 16, dim) + 5
        output = module(x)

        assert torch.allclose(output.mean(dim=-1), torch.zeros(4, 16), atol=1e-5), (
            "LayerNorm output mean is not close to zero"
        )
        # LayerNorm uses the biased variance; eps=1e-5 lowers the std by ~1e-6
        assert torch.allclose(
            output.std(dim=-1, unbiased=False), torch.ones(4, 16), atol=1e-4
        ), "LayerNorm output std is not close to one"


RESIDUAL_PROJECTIONS = (".context.out_proj.weight", ".feedforward.2.weight")


@pytest.mark.parametrize("init_std", [0.02, 0.05])
@pytest.mark.parametrize("tie", [True, False])
def test_weight_initialization(init_std, tie):
    """Parameters follow the GPT-2-style scheme in llamlam/model.py (DEC-002).

    Every weight (embed, qkv, feed-forward, untied head) ~ N(0, s^2) for
    s = init_std; the residual output projections (attention out_proj, last
    feed-forward linear) ~ N(0, s^2 / (2 * n_layers)); pos_embed = 0; biases = 0;
    norm gains = 1.
    Sizes keep every tensor at >= 4096 entries, so the std band (6 %) is >= 5
    standard errors wide and the residual scale (1/sqrt(6) here) is unmistakable.
    """
    # bias=True covers the bias branch
    config = Config(
        max_seq_length=64,
        vocab_size=256,
        n_layers=3,
        n_heads=4,
        dim_head=16,
        bias=True,
        init_std=init_std,
        tie_word_embeddings=tie,
    )
    model = GPTModel(config)
    residual_std = init_std / (2 * config.n_layers) ** 0.5

    seen = set()
    for name, param in model.named_parameters():
        seen.add(name)
        is_norm = ".norm_" in name or name.startswith("ln_f.")
        if name.endswith(".bias"):
            assert torch.all(param == 0), f"Bias {name} is not zero"
        elif is_norm:
            assert torch.all(param == 1), f"Norm gain {name} is not one"
        elif name == "pos_embed":
            assert torch.all(param == 0), "pos_embed is not zero"
        else:
            assert name.endswith(".weight"), f"Unexpected parameter {name}"
            std = residual_std if name.endswith(RESIDUAL_PROJECTIONS) else init_std
            assert param.std().item() == pytest.approx(std, rel=0.06), (
                f"Weight {name} has std {param.std().item():.5f}, expected {std:.5f}"
            )
            # mean within 5 standard errors of zero
            assert param.mean().abs() < 5 * std / param.numel() ** 0.5, (
                f"Weight {name} has high mean"
            )

    # every class of parameter was checked; the head is the embedding iff tied
    assert {"embed.weight", "pos_embed", "ln_f.weight", "ln_f.bias"} <= seen
    for i in range(config.n_layers):
        for suffix in RESIDUAL_PROJECTIONS + (
            ".context.qkv_proj.weight",
            ".feedforward.0.weight",
            ".feedforward.0.bias",
            ".norm_1.weight",
        ):
            assert f"blocks.{i}{suffix}" in seen
    assert ("head.weight" in seen) is not tie
    assert (model.head.weight is model.embed.weight) is tie


def test_overfitting_small_dataset(model):
    input_ids = torch.randint(0, 100, (2, 16))
    optimizer = Adam(model.parameters(), lr=1e-3)

    initial_loss = model(input_ids)["loss"].item()

    for _ in range(1000):
        optimizer.zero_grad()
        loss = model(input_ids)["loss"]
        loss.backward()
        optimizer.step()

    final_loss = model(input_ids)["loss"].item()

    assert final_loss < initial_loss, "Model failed to overfit small dataset"
    assert final_loss < 0.1, "Model did not achieve near-zero loss on small dataset"
