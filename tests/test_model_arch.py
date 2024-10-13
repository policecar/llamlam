import pytest
import torch
import torch.nn as nn

from llamlam.config import Config
from llamlam.model import GPTModel, Attention, Block


@pytest.fixture
def config():
    return Config(max_length=16, vocab_size=100, n_layer=2, n_head=2, head_width=8)


@pytest.fixture
def model(config):
    return GPTModel(config)


def test_model_structure(model, config):
    assert isinstance(model.embed, nn.Embedding)
    assert model.embed.num_embeddings == config.vocab_size
    assert model.embed.embedding_dim == config.n_embd

    assert isinstance(model.pos_embed, nn.Parameter)
    assert model.pos_embed.shape == (1, config.max_length, config.n_embd)

    assert isinstance(model.blocks, nn.Sequential)
    assert len(model.blocks) == config.n_layer

    assert isinstance(model.ln_f, nn.LayerNorm)
    assert model.ln_f.normalized_shape[0] == config.n_embd

    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == config.n_embd
    assert model.head.out_features == config.vocab_size


def test_attention_structure(config):
    attention = Attention(config.n_embd, config.n_head)

    assert isinstance(attention.qkv_proj, nn.Linear)
    assert attention.qkv_proj.in_features == config.n_embd
    assert attention.qkv_proj.out_features == 3 * config.n_embd

    assert isinstance(attention.out_proj, nn.Linear)
    assert attention.out_proj.in_features == config.n_embd
    assert attention.out_proj.out_features == config.n_embd


def test_block_structure(config):
    block = Block(config)

    assert isinstance(block.attention, Attention)
    assert isinstance(block.feedforward, nn.Sequential)
    assert len(block.feedforward) == 3
    assert isinstance(block.feedforward[0], nn.Linear)
    assert isinstance(block.feedforward[1], nn.Module)  # GELU
    assert isinstance(block.feedforward[2], nn.Linear)

    assert isinstance(block.norm_1, nn.LayerNorm)
    assert isinstance(block.norm_2, nn.LayerNorm)


def test_model_output_shape(model):
    batch_size, seq_len = 4, 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    output = model(input_ids)

    assert output["logits"].shape == (batch_size, seq_len, model.config.vocab_size)
    assert output["loss"].shape == ()


def test_attention_output_shape(config):
    attention = Attention(config.n_embd, config.n_head)
    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, config.n_embd)

    output = attention(x)

    assert output.shape == (batch_size, seq_len, config.n_embd)


def test_block_output_shape(config):
    block = Block(config)
    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, config.n_embd)

    output = block(x)

    assert output.shape == (batch_size, seq_len, config.n_embd)


# def test_model_parameter_count(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     expected_params = (
#         model.config.vocab_size * model.config.n_embd +  # embedding
#         model.config.max_length * model.config.n_embd +  # positional embedding
#         model.config.n_layer * (
#             3 * model.config.n_embd * model.config.n_embd +  # attention qkv
#             model.config.n_embd * model.config.n_embd +  # attention out
#             4 * model.config.n_embd * model.config.n_embd +  # ff 1
#             4 * model.config.n_embd * model.config.n_embd +  # ff 2
#             4 * model.config.n_embd +  # layer norms
#         ) +
#         model.config.n_embd * model.config.vocab_size  # output head
#     )
#     assert total_params == expected_params, f"Expected {expected_params} parameters, but got {total_params}"
