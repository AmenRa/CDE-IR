from typing import Callable, Union


class EvalCollator:
    def __init__(
        self,
        query_tokenizer: Union[object, Callable],
        doc_tokenizer: Union[object, Callable],
    ):
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

    def pad_doc_ids(self, doc_ids: list[str], max_len: int) -> list[str]:
        """Pads Document IDs to specified length, appending `fake_doc` value.

        Args:
            doc_ids (list[str]): Document IDs.
            max_len (int): Maximum length.

        Returns:
            list[str]: Padded Document IDs.
        """
        return doc_ids + ["fake_doc"] * (max_len - len(doc_ids))

    def pad_batch_doc_ids(self, batch_doc_ids: list[list[str]]) -> list[list[str]]:
        """Pads a batch of Document IDs to maximum length found in the batch.

        Args:
            batch_doc_ids (list[list[str]]): Batch of Document IDs.

        Returns:
            list[list[str]]: Padded batch of Document IDs.
        """
        max_len = max((len(x) for x in batch_doc_ids))
        return [self.pad_doc_ids(doc_ids, max_len) for doc_ids in batch_doc_ids]

    def __call__(self, batch: list[str]):
        """Call method."""

        batch_query_id = [x["query_id"] for x in batch]
        batch_query = [x["query"] for x in batch]

        batch_doc_ids = self.pad_batch_doc_ids([x["doc_ids"] for x in batch])
        batch_docs = [x["docs"] for x in batch]

        # Encode texts ---------------------------------------------------------
        encoded_queries = self.query_tokenizer(batch_query)
        encoded_docs = self.doc_tokenizer([y for x in batch_docs for y in x])

        return (
            batch_query_id,
            batch_doc_ids,
            encoded_queries,
            encoded_docs,
        )
