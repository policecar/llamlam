"""Generation tests using an offline DummyTokenizer (no HF hub access)."""

import torch


def test_greedy_is_deterministic(model, dummy_tokenizer):
    a = model.generate(dummy_tokenizer, "x", max_new_tokens=10)
    b = model.generate(dummy_tokenizer, "x", max_new_tokens=10)
    assert a == b
    assert isinstance(a, str)


def test_greedy_appends_tokens(model, dummy_tokenizer):
    out = model.generate(dummy_tokenizer, "x", max_new_tokens=5)
    # prompt encodes to 3 ids; 5 new -> 8 space-separated ids
    assert len(out.split()) == 8


def test_sampling_runs_with_topk_and_topp(model, dummy_tokenizer):
    torch.manual_seed(0)
    out = model.generate(
        dummy_tokenizer,
        "x",
        max_new_tokens=5,
        do_sample=True,
        temperature=0.8,
        top_k=10,
        top_p=0.95,
    )
    assert len(out.split()) == 8


def test_sampling_differs_from_greedy(model, dummy_tokenizer):
    greedy = model.generate(dummy_tokenizer, "x", max_new_tokens=15)
    torch.manual_seed(1)
    sampled = model.generate(
        dummy_tokenizer, "x", max_new_tokens=15, do_sample=True, temperature=1.5
    )
    # Very likely different under high temperature on a random-init model.
    assert greedy != sampled
