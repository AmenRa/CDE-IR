from typing import Callable, Union


class TrainCollator:
    def __init__(
        self,
        query_tokenizer: Union[object, Callable],
        doc_tokenizer: Union[object, Callable],
    ):
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

    def __call__(self, batch: list[str]):
        batch_query = [x["query"] for x in batch]
        batch_pos_doc = [x["pos_doc"] for x in batch]
        batch_neg_doc = [x["neg_doc"] for x in batch]

        # Encode texts ---------------------------------------------------------
        encoded_queries = self.query_tokenizer(batch_query)
        encoded_docs = self.doc_tokenizer(batch_pos_doc + batch_neg_doc)

        return encoded_queries, encoded_docs
