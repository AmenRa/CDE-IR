from src.paths import datasets_path


def msmarco_passage_path():
    return datasets_path() / "msmarco-passage"


def collection_path():
    return msmarco_passage_path() / "collection.jsonl"


# Train ------------------------------------------------------------------------
def train_set_path():
    return msmarco_passage_path() / "train"


def train_queries_path():
    return train_set_path() / "queries.jsonl"


def train_triples_path():
    return train_set_path() / "triples.tsv"


def train_relevants_path():
    return train_set_path() / "relevants.jsonl"


def train_negatives_path():
    return train_set_path() / "negatives.jsonl"


# Val --------------------------------------------------------------------------
def val_set_path():
    return msmarco_passage_path() / "val"


def val_qrels_path():
    return val_set_path() / "qrels.json"


def val_queries_path():
    return val_set_path() / "queries.jsonl"


# Dev --------------------------------------------------------------------------
def dev_set_path():
    return msmarco_passage_path() / "dev"


def dev_qrels_path():
    return dev_set_path() / "qrels.json"


def dev_queries_path():
    return dev_set_path() / "queries.jsonl"


def dev_bm25_doc_ids_path():
    return dev_set_path() / "bm25_doc_ids.json"


# TREC DL 2019 -----------------------------------------------------------------
def trec_dl_2019_path():
    return msmarco_passage_path() / "trec_dl_2019"


def trec_dl_2019_qrels_path():
    return trec_dl_2019_path() / "qrels.json"


def trec_dl_2019_queries_path():
    return trec_dl_2019_path() / "queries.jsonl"


def trec_dl_2019_bm25_doc_ids_path():
    return trec_dl_2019_path() / "bm25_doc_ids.json"


# TREC DL 2020 -----------------------------------------------------------------
def trec_dl_2020_path():
    return msmarco_passage_path() / "trec_dl_2020"


def trec_dl_2020_qrels_path():
    return trec_dl_2020_path() / "qrels.json"


def trec_dl_2020_queries_path():
    return trec_dl_2020_path() / "queries.jsonl"


def trec_dl_2020_bm25_doc_ids_path():
    return trec_dl_2020_path() / "bm25_doc_ids.json"


# TREC DL HARD -----------------------------------------------------------------
def trec_dl_hard_path():
    return msmarco_passage_path() / "trec_dl_hard"


def trec_dl_hard_qrels_path():
    return trec_dl_hard_path() / "qrels.json"


def trec_dl_hard_queries_path():
    return trec_dl_hard_path() / "queries.jsonl"


def trec_dl_hard_bm25_doc_ids_path():
    return trec_dl_hard_path() / "bm25_doc_ids.json"
