import json
import os


class Tokenizer:
    def __init__(self, vocab_file, unk_token, bos_token, eos_token):
        self.special_tokens = [unk_token, bos_token, eos_token]
        self.vocab_file = vocab_file

    def encode(self, text):
        pass

    def decode(self, token_ids):
        pass

    def save_vocabulary(self, save_directory: str):
        pass

    def save_pretrained(self, save_directory):
        """Save the tokenizer configuration to the specified directory."""
        os.makedirs(save_directory, exist_ok=True)
        tokenizer_config = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        pass

    @property
    def default_chat_template(self):
        pass
