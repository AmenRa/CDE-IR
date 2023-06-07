from typing import Callable, Union


class CrossEvalCollator:
    def __init__(self, tokenizer: Union[object, Callable]):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """Call method."""

        batch_query_id = [x["query_id"] for x in batch]
        batch_query = [x["query"] for x in batch]

        batch_doc_ids = [x["doc_ids"] for x in batch]
        batch_docs = [x["docs"] for x in batch]

        # Encode texts ---------------------------------------------------------
        batch_tokens = []
        for query, docs in zip(batch_query, batch_docs):
            tokens = self.tokenizer([query] * len(docs), docs)
            batch_tokens.append(tokens)

        return batch_query_id, batch_doc_ids, batch_tokens
