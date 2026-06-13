"""Tests for the padding-aware loss and attention masking (regression coverage
for the bug where the loss was computed over pad tokens)."""

import torch

from llamlam.data import DataCollator


class _CharTokenizer:
    """Tiny deterministic tokenizer so the collator test needs no network.

    Emits ids per word and pads with id 0; mimics the HF __call__ contract used
    by DataCollator (returns a dict-like with input_ids / attention_mask).
    """

    pad_token_id = 0

    def __call__(
        self,
        texts,
        padding="longest",
        max_length=None,
        truncation=True,
        pad_to_multiple_of=None,
        return_tensors="pt",
    ):
        seqs = [[len(w) for w in t.split()] for t in texts]
        width = max(len(s) for s in seqs)
        if pad_to_multiple_of:
            width = -(-width // pad_to_multiple_of) * pad_to_multiple_of
        input_ids, attention_mask = [], []
        for s in seqs:
            pad = width - len(s)
            input_ids.append(s + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(s) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class _Cfg:
    max_seq_length = 32
    pad_to_multiple_of = None


def test_collator_masks_padding_in_labels():
    collate = DataCollator(_CharTokenizer(), _Cfg())
    out = collate([{"text": "a bb ccc dddd"}, {"text": "a bb"}])
    assert "labels" in out
    # Where attention_mask is 0, labels must be -100; elsewhere == input_ids.
    pad = out["attention_mask"] == 0
    assert (out["labels"][pad] == -100).all()
    keep = ~pad
    assert (out["labels"][keep] == out["input_ids"][keep]).all()


def test_padding_does_not_change_unmasked_loss(model):
    """Loss with padding masked must equal loss on the unpadded sequence."""
    model.eval()
    torch.manual_seed(0)
    real = torch.randint(1, model.config.vocab_size, (1, 8))

    # Pad on the right to length 12; mask the pad in attention + labels.
    pad = torch.zeros(1, 4, dtype=torch.long)
    padded = torch.cat([real, pad], dim=1)
    attn = torch.cat([torch.ones(1, 8), torch.zeros(1, 4)], dim=1)
    labels = padded.clone()
    labels[attn == 0] = -100

    with torch.no_grad():
        loss_unpadded = model(real, labels=real)["loss"]
        loss_padded = model(padded, attention_mask=attn, labels=labels)["loss"]

    assert torch.allclose(loss_unpadded, loss_padded, atol=1e-4)


def test_attention_mask_changes_logits(model, batch):
    model.eval()
    with torch.no_grad():
        masked = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        unmasked = model(batch["input_ids"])
    assert not torch.allclose(masked["logits"], unmasked["logits"])


def test_gradient_accumulation_equivalence(model, batch):
    """Summing grads over two half-batches == grad over the full batch.

    Cross-entropy averages over *valid* (non-ignored) target tokens, so each
    micro-batch is weighted by its share of valid tokens, not by sequence count.
    """
    model.eval()  # disable dropout for a deterministic comparison
    ids, labels = batch["input_ids"], batch["labels"]
    valid = labels[..., 1:] != -100

    model.zero_grad()
    model(ids, labels=labels)["loss"].backward()
    full = [p.grad.clone() for p in model.parameters()]

    half = ids.shape[0] // 2
    total = valid.sum().item()
    w1 = valid[:half].sum().item() / total
    w2 = valid[half:].sum().item() / total

    model.zero_grad()
    (model(ids[:half], labels=labels[:half])["loss"] * w1).backward()
    (model(ids[half:], labels=labels[half:])["loss"] * w2).backward()
    accum = [p.grad.clone() for p in model.parameters()]

    for g_full, g_accum in zip(full, accum):
        assert torch.allclose(g_full, g_accum, atol=1e-4)
