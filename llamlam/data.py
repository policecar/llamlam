import logging

logger = logging.getLogger(__name__)

MAX_VALIDATION_EXAMPLES = 2560
VALIDATION_FRACTION = 0.1
MIN_TOKENS = 2  # one input token plus one target
FILTER_BATCH_SIZE = 1000


# @decision DEC-002: pad each batch to its longest example (dynamic padding), not to
#   max_seq_length
# @rationale: padded positions cost a full forward/backward but carry no target
#   (the attention_mask excludes them from the loss, see model.causal_lm_loss), so
#   short batches no longer pay for max_seq_length. Attention is causal and padding
#   is on the right, so real positions are unaffected by how much padding follows:
#   loss and evaluate() are invariant to the padded length (tests/test_data.py).
class DataCollator:
    """
    Tokenize a batch of {"text": ...} examples into input_ids and attention_mask.

    Each example is truncated to config.max_seq_length tokens, and the batch is
    right-padded to its longest example, rounded up to a multiple of
    config.pad_to_multiple_of when that is set (the tokenizer raises if
    max_seq_length is not a multiple of it). No labels: the model derives its
    targets from input_ids and masks padded ones with the attention_mask.
    """

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, examples):
        texts = [example["text"] for example in examples]
        return self.tokenizer(
            texts,
            padding="longest",
            max_length=self.config.max_seq_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            truncation=True,
            return_tensors="pt",
        )


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


# @decision DEC-001: drop examples with fewer than MIN_TOKENS tokens when the dataset
#   is prepared, counting tokens exactly as DataCollator does (truncated to
#   max_seq_length, special tokens included)
# @rationale: such an example has no next-token target, and a micro-batch made only
#   of them makes causal_lm_loss raise. Filtering once up front keeps that error a
#   backstop instead of turning it into a silent zero loss; counting like the
#   collator means "kept" is exactly "has at least one real target in its batch".
def drop_short_examples(dataset, tokenizer, max_seq_length, name="dataset"):
    """
    Return `dataset` without examples whose "text" tokenizes to < MIN_TOKENS tokens.

    Logs how many examples were dropped; raises ValueError if none are left.
    """

    def long_enough(batch):
        input_ids = tokenizer(
            batch["text"],
            max_length=max_seq_length,
            truncation=True,
            return_attention_mask=False,
        )["input_ids"]
        return [len(ids) >= MIN_TOKENS for ids in input_ids]

    n_before = len(dataset)
    filtered = dataset.filter(long_enough, batched=True, batch_size=FILTER_BATCH_SIZE)
    n_dropped = n_before - len(filtered)
    logger.info(
        f"{name}: dropped {n_dropped} of {n_before} examples with fewer than "
        f"{MIN_TOKENS} tokens"
    )
    if len(filtered) == 0:
        raise ValueError(
            f"{name}: all {n_before} examples have fewer than {MIN_TOKENS} tokens "
            f"(max_seq_length={max_seq_length}); nothing left to train or evaluate on"
        )
    return filtered
