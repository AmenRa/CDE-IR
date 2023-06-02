from transformers import AutoTokenizer


class QueryTokenizer:
    def __init__(
        self,
        language_model: str = "bert-base-uncased",
        max_len: int = 32,
        add_prefix_token: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model, use_fast=True
        )

        self.max_len = max_len

        self.add_prefix_token = add_prefix_token
        self.prefix_token = "[unused0]"
        self.prefix_token_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[unused0]"]}
        )

    def __call__(self, texts: list[str]) -> dict:
        assert type(texts) == list and all(
            type(x) == str for x in texts
        ), "Error: `texts` should be a list of strings."

        if self.add_prefix_token:
            texts = [f"{self.prefix_token} {x}" for x in texts]

        out = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        out["input_ids"][
            out["input_ids"] == self.tokenizer.pad_token_id
        ] = self.tokenizer.mask_token_id

        return out
