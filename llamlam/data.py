from datasets import load_dataset
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, tokenizer, type_path="train", max_length=512, debug=False):
        if debug:
            vernum = 2
        else:
            vernum = 103
        self.vernum = vernum
        self.dataset = load_dataset(
            "wikitext", f"wikitext-{vernum}-raw-v1", split=type_path
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return int(len(self.dataset) * 0.1) if (self.vernum == 103) else 32  # type: ignore

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]  # type: ignore
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.squeeze()}
