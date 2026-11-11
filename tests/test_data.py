"""DataCollator padding branches (item 8) and validation split (item 9)."""

import math

import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from llamlam.config import Config
from llamlam.data import MAX_VALIDATION_EXAMPLES, DataCollator, train_val_split

TEXTS = [
    {"text": "Once upon a time there was a small language model."},
    {"text": "Hi."},
    {"text": "It learned to count tokens one by one."},
]


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def real_lengths(tokenizer, examples):
    return [len(tokenizer(e["text"])["input_ids"]) for e in examples]


def test_pad_to_multiple_pads_to_rounded_longest(tokenizer):
    config = Config(max_seq_length=32, pad_to_multiple_of=8)
    batch = DataCollator(tokenizer, config)(TEXTS)
    lengths = real_lengths(tokenizer, TEXTS)
    expected_len = math.ceil(max(lengths) / 8) * 8
    assert expected_len < config.max_seq_length  # distinguishes from max_length padding
    assert batch["input_ids"].shape == (len(TEXTS), expected_len)
    assert batch["attention_mask"].sum(dim=1).tolist() == lengths


def test_pad_to_multiple_truncates_to_max_seq_length(tokenizer):
    config = Config(max_seq_length=16, pad_to_multiple_of=8)
    long_text = [{"text": " ".join(["word"] * 100)}]
    batch = DataCollator(tokenizer, config)(long_text)
    assert batch["input_ids"].shape == (1, 16)


def test_pad_to_multiple_rejects_incompatible_max_seq_length(tokenizer):
    config = Config(max_seq_length=20, pad_to_multiple_of=8)
    with pytest.raises(ValueError):
        DataCollator(tokenizer, config)(TEXTS)


def test_default_pads_to_max_seq_length(tokenizer):
    config = Config(max_seq_length=24)
    batch = DataCollator(tokenizer, config)(TEXTS)
    assert batch["input_ids"].shape == (len(TEXTS), 24)
    assert batch["attention_mask"].sum(dim=1).tolist() == real_lengths(tokenizer, TEXTS)
    pad_positions = batch["attention_mask"] == 0
    assert (batch["input_ids"][pad_positions] == tokenizer.eos_token_id).all()


def make_dataset(n):
    return DatasetDict(
        {"train": Dataset.from_dict({"text": [str(i) for i in range(n)]})}
    )


@pytest.mark.parametrize(
    "n_examples, expected_val",
    [(50, 5), (5, 1), (2, 1), (30000, MAX_VALIDATION_EXAMPLES)],
)
def test_validation_split_sizes(n_examples, expected_val):
    train, val = train_val_split(make_dataset(n_examples))
    assert len(val) == expected_val
    assert len(train) == n_examples - expected_val
    assert set(train["text"]).isdisjoint(val["text"])


def test_validation_split_too_small_raises():
    with pytest.raises(ValueError, match="at least 2"):
        train_val_split(make_dataset(1))


def test_existing_validation_split_is_used():
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["a", "b", "c"]}),
            "validation": Dataset.from_dict({"text": ["v"]}),
            "test": Dataset.from_dict({"text": ["t"]}),
        }
    )
    train, val = train_val_split(dataset)
    assert train["text"] == ["a", "b", "c"]
    assert val["text"] == ["v"]
