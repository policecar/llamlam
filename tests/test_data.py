"""
DataCollator dynamic padding (item 8), validation split (item 9) and dropping
examples without a next-token target.
"""

import logging
import math

import pytest
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import llamlam.data
from llamlam.config import Config
from llamlam.data import (
    MAX_VALIDATION_EXAMPLES,
    DataCollator,
    drop_short_examples,
    train_val_split,
)
from llamlam.difftransformer import DiffTransformer
from llamlam.model import GPTModel
from llamlam.utils import evaluate

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


def test_default_pads_to_longest_example(tokenizer):
    config = Config(max_seq_length=24)
    batch = DataCollator(tokenizer, config)(TEXTS)
    lengths = real_lengths(tokenizer, TEXTS)
    assert max(lengths) < config.max_seq_length  # distinguishes from max_length padding
    assert batch["input_ids"].shape == (len(TEXTS), max(lengths))
    assert batch["attention_mask"].sum(dim=1).tolist() == lengths
    pad_positions = batch["attention_mask"] == 0
    assert pad_positions.any()  # the short example is padded
    assert (batch["input_ids"][pad_positions] == tokenizer.eos_token_id).all()
    # right padding: each row is its real tokens, then padding
    for row, n in zip(batch["attention_mask"], lengths):
        assert row[:n].all() and not row[n:].any()


def test_collator_returns_only_input_ids_and_attention_mask(tokenizer):
    for pad_to_multiple_of in (None, 8):
        config = Config(max_seq_length=32, pad_to_multiple_of=pad_to_multiple_of)
        batch = DataCollator(tokenizer, config)(TEXTS)
        assert set(batch.keys()) == {"input_ids", "attention_mask"}


def test_default_truncates_to_max_seq_length(tokenizer):
    config = Config(max_seq_length=10)
    examples = [{"text": " ".join(["word"] * 100)}, {"text": "Hi."}]
    batch = DataCollator(tokenizer, config)(examples)
    assert batch["input_ids"].shape == (2, 10)
    assert batch["attention_mask"].sum(dim=1).tolist() == [10, 2]


def test_batch_length_varies_with_its_examples(tokenizer):
    config = Config(max_seq_length=64)
    collate = DataCollator(tokenizer, config)
    short = collate(TEXTS[1:2])  # "Hi."
    long = collate(TEXTS)
    assert short["input_ids"].shape == (1, real_lengths(tokenizer, TEXTS[1:2])[0])
    assert long["input_ids"].shape[1] == max(real_lengths(tokenizer, TEXTS))


##########################################
# Loss and evaluation do not depend on the padded length
##########################################


def max_length_batch(tokenizer, config, examples):
    """The pre-dynamic-padding collator output: every row padded to max_seq_length."""
    return tokenizer(
        [e["text"] for e in examples],
        padding="max_length",
        max_length=config.max_seq_length,
        truncation=True,
        return_tensors="pt",
    )


@pytest.mark.parametrize("model_cls", [GPTModel, DiffTransformer])
def test_loss_and_eval_invariant_to_padding_amount(tokenizer, model_cls):
    config = Config(
        max_seq_length=32,
        vocab_size=len(tokenizer),
        n_layers=2,
        n_heads=2,
        dim_head=8,
        dropout=0.0,
    )
    torch.manual_seed(0)
    model = model_cls(config).eval()
    dynamic = DataCollator(tokenizer, config)(TEXTS)
    padded = max_length_batch(tokenizer, config, TEXTS)
    assert dynamic["input_ids"].shape[1] < padded["input_ids"].shape[1] == 32
    # same real tokens, only the padding differs
    n = dynamic["input_ids"].shape[1]
    assert (padded["input_ids"][:, :n] == dynamic["input_ids"]).all()
    assert not padded["attention_mask"][:, n:].any()

    def loss(batch):
        with torch.no_grad():
            out = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        return out["loss"].item()

    assert loss(dynamic) == pytest.approx(loss(padded), rel=1e-5)
    # evaluate() over several batches of different padded lengths
    batches_dynamic = [DataCollator(tokenizer, config)([e]) for e in TEXTS]
    batches_padded = [max_length_batch(tokenizer, config, [e]) for e in TEXTS]
    assert len({b["input_ids"].shape[1] for b in batches_dynamic}) > 1
    loss_dynamic, ppl_dynamic = evaluate(model, batches_dynamic)
    loss_padded, ppl_padded = evaluate(model, batches_padded)
    assert loss_dynamic == pytest.approx(loss_padded, rel=1e-5)
    assert ppl_dynamic == pytest.approx(ppl_padded, rel=1e-5)


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


##########################################
# Examples with fewer than two tokens
##########################################

SHORT = ["", " ", "\n", "Hi", "hello"]  # 0 or 1 GPT-2 tokens: no target
LONG_ENOUGH = ["Hi.", "Hello world", "a b"]  # exactly 2 tokens
LONG = " ".join(["word"] * 50)


def test_token_counts_premise(tokenizer):
    # the filter tests below rely on these exact GPT-2 token counts
    assert [len(tokenizer(t)["input_ids"]) for t in SHORT] == [0, 1, 1, 1, 1]
    assert [len(tokenizer(t)["input_ids"]) for t in LONG_ENOUGH] == [2, 2, 2]
    assert len(tokenizer(LONG)["input_ids"]) == 50


def mixed_texts():
    # short texts interleaved with kept ones, so batching cannot hide a mistake
    return [SHORT[0], LONG_ENOUGH[0], SHORT[1], SHORT[2], LONG, SHORT[3]] + [
        LONG_ENOUGH[1],
        SHORT[4],
        LONG_ENOUGH[2],
    ]


@pytest.mark.parametrize("max_seq_length", [1024, 2])  # truncation inactive / active
def test_drop_short_examples_keeps_two_token_texts(
    tokenizer, max_seq_length, monkeypatch
):
    monkeypatch.setattr(llamlam.data, "FILTER_BATCH_SIZE", 4)  # several batches
    dataset = Dataset.from_dict({"text": mixed_texts()})
    kept = drop_short_examples(dataset, tokenizer, max_seq_length)
    assert kept["text"] == [LONG_ENOUGH[0], LONG, LONG_ENOUGH[1], LONG_ENOUGH[2]]


def test_drop_short_examples_counts_tokens_after_truncation(tokenizer):
    # with max_seq_length=1 the collator leaves every example a single token
    dataset = Dataset.from_dict({"text": [LONG, LONG_ENOUGH[0]]})
    with pytest.raises(ValueError, match="all 2 examples have fewer than 2 tokens"):
        drop_short_examples(dataset, tokenizer, max_seq_length=1)


def test_drop_short_examples_raises_on_all_short_split(tokenizer):
    dataset = Dataset.from_dict({"text": SHORT})
    with pytest.raises(ValueError, match="validation: all 5 examples"):
        drop_short_examples(dataset, tokenizer, 1024, name="validation")


def test_drop_short_examples_logs_dropped_count(tokenizer, caplog):
    dataset = Dataset.from_dict({"text": mixed_texts()})
    with caplog.at_level(logging.INFO, logger="llamlam.data"):
        drop_short_examples(dataset, tokenizer, 1024, name="train")
    assert "train: dropped 5 of 9 examples" in caplog.text


def n_targets_per_example(dataset, tokenizer, config):
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=DataCollator(tokenizer, config)
    )
    return [
        n for batch in loader for n in batch["attention_mask"][:, 1:].sum(1).tolist()
    ]


@pytest.mark.parametrize("pad_to_multiple_of", [None, 8])
def test_collated_batches_have_a_target_per_example(tokenizer, pad_to_multiple_of):
    config = Config(max_seq_length=16, pad_to_multiple_of=pad_to_multiple_of)
    dataset = Dataset.from_dict({"text": mixed_texts()})
    # without filtering, some examples (and a whole batch) have no target
    unfiltered = n_targets_per_example(dataset, tokenizer, config)
    assert unfiltered.count(0) == len(SHORT)
    assert unfiltered[2:4] == [0, 0]  # batch 1 = two short texts: loss would raise

    kept = drop_short_examples(dataset, tokenizer, config.max_seq_length)
    targets = n_targets_per_example(kept, tokenizer, config)
    assert len(targets) == len(kept) == 4
    assert min(targets) >= 1
