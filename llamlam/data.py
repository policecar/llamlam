class DataCollator:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, examples):
        examples = [example["text"] for example in examples]
        if self.config.pad_to_multiple_of is not None:
            batch = self.tokenizer(
                examples,
                padding="True",
                max_length=None,
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
