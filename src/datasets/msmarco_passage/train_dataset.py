import random

from indxr import Indxr
from torch.utils.data import Dataset
from unified_io import read_list

from .paths import *


class TrainDataset(Dataset):
    def __init__(self, kind: str = "triples", n_samples: int = 0):
        assert kind in {"triples", "negatives"}

        self.queries_index = Indxr(train_queries_path())
        self.doc_index = Indxr(collection_path())

        if kind == "triples":
            self.train_triples = read_list(train_triples_path())
            if n_samples > 0:
                self.train_triples = random.sample(self.train_triples, n_samples)

        elif kind == "negatives":
            self.relevants_index = Indxr(train_relevants_path())
            self.negatives_index = Indxr(train_negatives_path())

    def get_query(self, q_id):
        return self.queries_index.get(q_id)["text"]

    def get_documents(self, pos_doc_id, neg_doc_id):
        pos_doc = self.doc_index.get(pos_doc_id)["text"]
        neg_doc = (
            self.doc_index.get(neg_doc_id)["text"]
            if neg_doc_id != "fake_doc"
            else "[PAD]"
        )

        return pos_doc, neg_doc

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index: int) -> str:
        q_id, pos_doc_id, neg_doc_id = self.train_triples[index].split("\t")

        query = self.get_query(q_id)
        pos_doc, neg_doc = self.get_documents(pos_doc_id, neg_doc_id)

        return {"query": query, "pos_doc": pos_doc, "neg_doc": neg_doc}

    # This allows to call len(dataset) to get the dataset size
    def __len__(self) -> int:
        return len(self.train_triples)
