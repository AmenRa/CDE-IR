from indxr import Indxr
from torch.utils.data import Dataset
from unified_io import read_json

from .paths import *


class EvalDataset(Dataset):
    def __init__(self, split: str):
        assert split in {"dev", "trec-dl-2019", "trec-dl-2020"}

        self.doc_index = Indxr(collection_path())

        if split == "dev":
            self.queries_index = Indxr(dev_queries_path())
            self.doc_ids = read_json(dev_bm25_doc_ids_path())

        elif split == "trec-dl-2019":
            self.queries_index = Indxr(trec_dl_2019_queries_path())
            self.doc_ids = read_json(trec_dl_2019_bm25_doc_ids_path())

        elif split == "trec-dl-2020":
            self.queries_index = Indxr(trec_dl_2020_queries_path())
            self.doc_ids = read_json(trec_dl_2020_bm25_doc_ids_path())

    def get_documents(self, doc_ids):
        docs = self.doc_index.mget(doc_ids)
        return [x["text"] for x in docs]

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index: int) -> str:
        query = self.queries_index[index]
        doc_ids = self.doc_ids[query["id"]][:1_000]
        docs = self.get_documents(doc_ids)

        if len(docs) < 1_000:
            docs += ["[PAD]"] * (1_000 - len(docs))

        return {
            "query": query["text"],
            "rel_doc_ids": query.get("rel_doc_ids", []),
            "docs": docs,
            "query_id": query["id"],
            "doc_ids": doc_ids,
        }

    # This allows to call len(dataset) to get the dataset size
    def __len__(self) -> int:
        return len(self.queries_index)
