import contextlib
from collections import defaultdict
from pathlib import Path

import ir_datasets
from oneliner_utils import write_json, write_jsonl, write_list
from tqdm import tqdm

datasets_path = Path("datasets")
msmarco_passage_path = datasets_path / "msmarco_passage"
collection_path = msmarco_passage_path / "collection.jsonl"

# Train
train_set_path = msmarco_passage_path / "train"
train_queries_path = train_set_path / "queries.jsonl"
train_triples_path = train_set_path / "triples.tsv"

# Val
val_set_path = msmarco_passage_path / "val"
val_qrels_path = val_set_path / "qrels.json"
val_queries_path = val_set_path / "queries.jsonl"

# Dev
dev_set_path = msmarco_passage_path / "dev"
dev_qrels_path = dev_set_path / "qrels.json"
dev_queries_path = dev_set_path / "queries.jsonl"
dev_bm25_doc_ids_path = dev_set_path / "bm25_doc_ids.json"

# TREC DL 2019
trec_dl_2019_path = msmarco_passage_path / "trec_dl_2019"
trec_dl_2019_qrels_path = trec_dl_2019_path / "qrels.json"
trec_dl_2019_queries_path = trec_dl_2019_path / "queries.jsonl"
trec_dl_2019_bm25_doc_ids_path = trec_dl_2019_path / "bm25_doc_ids.json"

# TREC DL 2020
trec_dl_2020_path = msmarco_passage_path / "trec_dl_2020"
trec_dl_2020_qrels_path = trec_dl_2020_path / "qrels.json"
trec_dl_2020_queries_path = trec_dl_2020_path / "queries.jsonl"
trec_dl_2020_bm25_doc_ids_path = trec_dl_2020_path / "bm25_doc_ids.json"

# TREC DL HARD
trec_dl_hard_path = msmarco_passage_path / "trec_dl_hard"
trec_dl_hard_qrels_path = trec_dl_hard_path / "qrels.json"
trec_dl_hard_queries_path = trec_dl_hard_path / "queries.jsonl"
trec_dl_hard_bm25_doc_ids_path = trec_dl_hard_path / "bm25_doc_ids.json"


def download_collection():
    if not collection_path.exists():
        dataset = ir_datasets.load("msmarco-passage")

        write_jsonl(
            tqdm(
                dataset.docs_iter(),
                total=dataset.docs_count(),
                dynamic_ncols=True,
                desc="Exporting collection",
            ),
            path=collection_path,
            callback=lambda x: {"id": str(x.doc_id), "text": x.text},
        )


def download_train():
    if not train_queries_path.exists():
        dataset = ir_datasets.load("msmarco-passage/train/judged")

        train_qrels = defaultdict(dict)
        for q in dataset.qrels_iter():
            train_qrels[str(q.query_id)][str(q.doc_id)] = q.relevance

        write_jsonl(
            dataset.queries_iter(),
            path=train_queries_path,
            callback=lambda x: {
                "id": str(x.query_id),
                "text": x.text,
                "rel_doc_ids": list(train_qrels[str(x.query_id)]),
            },
        )


def download_train_triples():
    if not train_triples_path.exists():
        dataset = ir_datasets.load("msmarco-passage/train/triples-small")
        # dataset = ir_datasets.load("msmarco-passage/train/triples-v2")
        write_list(
            tqdm(
                dataset.docpairs_iter(),
                total=dataset.docpairs_count(),
                dynamic_ncols=True,
                desc="Exporting train triples",
            ),
            path=train_triples_path,
            callback=lambda x: f"{x.query_id}\t{x.doc_id_a}\t{x.doc_id_b}",
        )


def download_val():
    if not val_qrels_path.exists() or not val_queries_path.exists():
        dataset = ir_datasets.load("msmarco-passage/dev/small")
        dev_q_ids = {str(q.query_id) for q in dataset.queries_iter()}

        dataset = ir_datasets.load("msmarco-passage/dev/judged")
        val_q_ids = [str(q.query_id) for q in dataset.queries_iter()]
        val_q_ids = [q_id for q_id in val_q_ids if q_id not in dev_q_ids]

        print(len(val_q_ids))

        # qrels = defaultdict(dict)
        # for q in dataset.qrels_iter():
        #     qrels[str(q.query_id)][str(q.doc_id)] = q.relevance

        # write_json(qrels, dev_qrels_path)
        # write_jsonl(
        #     dataset.queries_iter(),
        #     path=dev_queries_path,
        #     callback=lambda x: {"id": str(x.query_id), "text": x.text},
        # )


def download_eval_data(ir_datasets_id, qrels_path, queries_path, bm25_doc_ids_path):
    if qrels_path.exists() and queries_path.exists() and bm25_doc_ids_path.exists():
        return

    dataset = ir_datasets.load(ir_datasets_id)

    qrels = defaultdict(dict)
    for q in dataset.qrels_iter():
        qrels[str(q.query_id)][str(q.doc_id)] = q.relevance

    # Save qrels
    write_json(qrels, qrels_path)

    # Save queries
    write_jsonl(
        dataset.queries_iter(),
        path=queries_path,
        callback=lambda x: {"id": str(x.query_id), "text": x.text},
    )

    with contextlib.suppress(Exception):
        # Save bm25 doc ids
        bm25_doc_ids = defaultdict(list)
        for x in dataset.scoreddocs_iter():
            bm25_doc_ids[x.query_id].append(x.doc_id)
        write_json(bm25_doc_ids, bm25_doc_ids_path)


def download_dev():
    download_eval_data(
        ir_datasets_id="msmarco-passage/dev/small",
        qrels_path=dev_qrels_path,
        queries_path=dev_queries_path,
        bm25_doc_ids_path=dev_bm25_doc_ids_path,
    )


def download_trec_dl_2019():
    download_eval_data(
        ir_datasets_id="msmarco-passage/trec-dl-2019/judged",
        qrels_path=trec_dl_2019_qrels_path,
        queries_path=trec_dl_2019_queries_path,
        bm25_doc_ids_path=trec_dl_2019_bm25_doc_ids_path,
    )


def download_trec_dl_2020():
    download_eval_data(
        ir_datasets_id="msmarco-passage/trec-dl-2020/judged",
        qrels_path=trec_dl_2020_qrels_path,
        queries_path=trec_dl_2020_queries_path,
        bm25_doc_ids_path=trec_dl_2020_bm25_doc_ids_path,
    )


def download_trec_dl_hard():
    download_eval_data(
        ir_datasets_id="msmarco-passage/trec-dl-hard",
        qrels_path=trec_dl_hard_qrels_path,
        queries_path=trec_dl_hard_queries_path,
        bm25_doc_ids_path=trec_dl_hard_bm25_doc_ids_path,
    )


def download_msmarco():
    download_collection()
    download_train()
    download_train_triples()
    download_val()
    download_dev()
    download_trec_dl_2019()
    download_trec_dl_2020()
    download_trec_dl_hard()
