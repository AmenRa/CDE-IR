from pathlib import Path

from indxr import Indxr
from oneliner_utils import join_path, read_json
from torch.utils.data import Dataset


class EvalDataset(Dataset):
    def __init__(
        self,
        split: str,
    ):
        datasets_path = Path("datasets")
        msmarco_passage_path = datasets_path / "msmarco_passage"
        split_path = msmarco_passage_path / split
        collection_path = msmarco_passage_path / "collection.jsonl"
        queries_path = split_path / "queries.jsonl"
        bm25_doc_ids_path = split_path / "bm25_doc_ids.json"

        self.queries_index = Indxr(queries_path)
        self.doc_index = Indxr(collection_path)
        self.bm25_doc_ids = read_json(bm25_doc_ids_path)

    def get_documents(self, bm25_doc_ids):
        docs = self.doc_index.mget(bm25_doc_ids)
        return [x["text"] for x in docs]

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index: int) -> str:
        query = self.queries_index[index]
        bm25_doc_ids = self.bm25_doc_ids[query["id"]][:1_000]
        docs = self.get_documents(bm25_doc_ids)

        if len(docs) < 1_000:
            docs += ["[PAD]"] * (1_000 - len(docs))

        return {
            "query": query["text"],
            "rel_doc_ids": query.get("rel_doc_ids", []),
            "docs": docs,
            "query_id": query["id"],
            "doc_ids": bm25_doc_ids,
        }

    # This allows to call len(dataset) to get the dataset size
    def __len__(self) -> int:
        return len(self.queries_index)
