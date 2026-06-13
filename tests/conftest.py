"""Shared fixtures for the llamlam test suite.

Everything here is network-free: models run on tiny synthetic token ids and a
DummyTokenizer stands in for gpt2 so the suite never touches the HF hub.
"""

import pytest
import torch

from llamlam.config import Config
from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel

MODEL_FACTORIES = {"gpt": GPTModel, "diff": DiffTransformer}


@pytest.fixture
def config():
    return Config(
        max_seq_length=32,
        vocab_size=64,
        n_layers=2,
        n_heads=2,
        dim_head=8,
        dropout=0.1,
    )


@pytest.fixture(params=["gpt", "diff"])
def model_name(request):
    return request.param


@pytest.fixture
def model(model_name, config):
    torch.manual_seed(0)
    return MODEL_FACTORIES[model_name](config)


@pytest.fixture
def batch(config):
    """A right-padded batch with labels masked (-100) at padding positions."""
    torch.manual_seed(0)
    batch_size, seq_len = 3, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len // 2 :] = 0  # pad the second half (right padding)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DummyTokenizer:
    """Minimal stand-in for an HF tokenizer, for offline generate tests."""

    def encode(self, text, return_tensors=None):
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids.tolist())


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()
