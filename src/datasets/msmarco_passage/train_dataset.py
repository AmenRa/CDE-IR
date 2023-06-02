import random
from pathlib import Path

from indxr import Indxr
from oneliner_utils import read_list
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, n_samples: int = 0):
        datasets_path = Path("datasets")
        msmarco_passage_path = datasets_path / "msmarco_passage"
        train_set_path = msmarco_passage_path / "train"
        collection_path = msmarco_passage_path / "collection.jsonl"
        queries_path = train_set_path / "queries.jsonl"
        triples_path = train_set_path / "triples.tsv"

        self.queries_index = Indxr(str(queries_path))
        self.doc_index = Indxr(str(collection_path))
        self.train_triples = read_list(triples_path)

        if n_samples > 0:
            self.train_triples = random.sample(self.train_triples, n_samples)

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
