MAX_VALIDATION_EXAMPLES = 2560
VALIDATION_FRACTION = 0.1


class DataCollator:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, examples):
        examples = [example["text"] for example in examples]
        if self.config.pad_to_multiple_of is not None:
            # Pad to the longest example in the batch, rounded up to a multiple of
            # pad_to_multiple_of; the tokenizer raises if max_seq_length is not a
            # multiple of it.
            batch = self.tokenizer(
                examples,
                padding=True,
                max_length=self.config.max_seq_length,
                pad_to_multiple_of=self.config.pad_to_multiple_of,
                truncation=True,
                return_tensors="pt",
            )
        else:
            batch = self.tokenizer(
                examples,
                padding="max_length",
                max_length=self.config.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
        return batch


def train_val_split(dataset):
    """
    Return (train, validation) splits of a DatasetDict.

    Uses the dataset's own "validation" split if it has one; otherwise splits off
    10% of "train" (at least 1, at most MAX_VALIDATION_EXAMPLES examples).
    """
    if "validation" in dataset:
        return dataset["train"], dataset["validation"]

    n_examples = len(dataset["train"])
    if n_examples < 2:
        raise ValueError(
            f"Need at least 2 training examples to split off a validation set, "
            f"got {n_examples}"
        )
    test_size = min(
        MAX_VALIDATION_EXAMPLES, max(1, int(VALIDATION_FRACTION * n_examples))
    )
    split = dataset["train"].train_test_split(test_size=test_size)
    return split["train"], split["test"]
