from transformers import AutoTokenizer


class QueryTokenizer:
    def __init__(
        self,
        language_model: str = "bert-base-uncased",
        max_len: int = 64,
        add_prefix_token: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(language_model, use_fast=True)

        self.max_len = max_len

        self.add_prefix_token = add_prefix_token
        self.prefix_token = "[unused0]"
        self.prefix_token_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]"]})

    def __call__(self, texts: list[str]) -> dict:
        if self.add_prefix_token:
            texts = [f"{self.prefix_token} {x}" for x in texts]

        return self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
            # return_attention_mask=
            # return_offsets_mapping=
            # return_length=
            # return_special_tokens_mask=
            return_token_type_ids=False,
        )
