class DataCollator:
    """Tokenize a batch of raw-text examples for causal language modeling.

    Produces ``input_ids``, ``attention_mask`` and ``labels``. ``labels`` is a
    copy of ``input_ids`` with padding positions set to ``-100`` so that the
    cross-entropy loss (``ignore_index=-100``) skips them. Sequences are padded
    dynamically to the longest example in the batch (optionally rounded up to
    ``pad_to_multiple_of`` for tensor-core friendliness), which is far cheaper
    than always padding to ``max_seq_length``.
    """

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, examples):
        texts = [example["text"] for example in examples]
        batch = self.tokenizer(
            texts,
            padding="longest",
            max_length=self.config.max_seq_length,
            truncation=True,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Labels mirror input_ids, but padding is masked out of the loss.
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
