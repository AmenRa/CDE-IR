from transformers import AutoTokenizer


class DocTokenizer:
    def __init__(
        self,
        encoder: str = "bert-base-uncased",
        max_len: int = 512,
        add_prefix_token: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)

        self.max_len = max_len

        self.add_prefix_token = add_prefix_token
        self.prefix_token = "[unused1]"
        self.prefix_token_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]"]})

    def __call__(self, texts: list[str]) -> dict:
        if self.add_prefix_token:
            texts = [f"{self.prefix_token} {x}" for x in texts]

        return self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
