from torch import Tensor
from transformers import AutoTokenizer

from tokenizers.pre_tokenizers import BertPreTokenizer


class CrossTokenizer:
    def __init__(
        self,
        language_model: str = "bert-base-uncased",
        query_max_len: int = 64,
        doc_max_len: int = 512,
    ):
        self.pre_tokenizer = BertPreTokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model, use_fast=True
        )
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len

    def truncate(self, texts: list[str], max_len: int):
        token_boundaries = [
            [x[1][1] for x in self.pre_tokenizer.pre_tokenize_str(text)]
            for text in texts
        ]
        truncation_indices = [
            [y for i, y in enumerate(x) if i < max_len][-1]
            for x in token_boundaries
        ]
        return [t[:i] for t, i in zip(texts, truncation_indices)]

    def __call__(self, queries: list[str], docs: list[str]) -> dict:
        return self.tokenizer(
            self.truncate(queries, self.query_max_len),
            self.truncate(docs, self.doc_max_len),
            max_len=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
