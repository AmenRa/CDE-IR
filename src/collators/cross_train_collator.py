from typing import Callable, Union


class CrossTrainCollator:
    def __init__(self, tokenizer: Union[object, Callable]):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]):
        batch_query = [x["query"] for x in batch]
        batch_pos_doc = [x["pos_doc"] for x in batch]
        batch_neg_doc = [x["neg_doc"] for x in batch]

        # Encode texts ---------------------------------------------------------
        pos_tokens = self.tokenizer(batch_query, batch_pos_doc)
        neg_tokens = self.tokenizer(batch_query, batch_neg_doc)

        return pos_tokens, neg_tokens
