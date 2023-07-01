import contextlib
import gzip
import pickle
from collections import defaultdict

import ir_datasets
from tqdm import tqdm
from unified_io import write_json, write_jsonl, write_list

from src.utils import download_file

from .paths import *


def download_collection():
    if not collection_path().exists():
        dataset = ir_datasets.load("msmarco-passage")

        write_jsonl(
            tqdm(
                dataset.docs_iter(),
                total=dataset.docs_count(),
                dynamic_ncols=True,
                desc="Exporting collection",
            ),
            path=collection_path(),
            callback=lambda x: {"id": str(x.doc_id), "text": x.text},
        )


def download_train_queries():
    if train_queries_path().exists():
        return

    dataset = ir_datasets.load("msmarco-passage/train/judged")

    write_jsonl(
        tqdm(
            dataset.queries_iter(),
            total=dataset.queries_count(),
            dynamic_ncols=True,
            desc="Exporting train queries",
        ),
        path=train_queries_path(),
        callback=lambda x: {"id": str(x.query_id), "text": x.text},
    )


def download_train_relevants_and_negatives():
    if train_relevants_path().exists() and train_negatives_path().exists():
        return

    dataset = ir_datasets.load("msmarco-passage/train/judged")

    # Extract relevants --------------------------------------------------------
    relevants_dict = defaultdict(list)
    for x in tqdm(
        dataset.qrels_iter(),
        total=dataset.qrels_count(),
        dynamic_ncols=True,
        desc="Exporting train queries relevants",
    ):
        relevants_dict[x.query_id].append(x.doc_id)

    relevants = [{"id": str(k), "doc_ids": v} for k, v in relevants_dict.items()]
    relevants = sorted(relevants, key=lambda x: x["id"])
    write_jsonl(relevants, train_relevants_path())

    # Extract negatives --------------------------------------------------------
    negatives_dict = defaultdict(list)
    for x in tqdm(
        dataset.scoreddocs_iter(),
        total=dataset.scoreddocs_count(),
        dynamic_ncols=True,
        desc="Exporting train queries negatives",
    ):
        if x.doc_id not in relevants_dict[x.query_id]:
            negatives_dict[x.query_id].append(x.doc_id)

    negatives = [{"id": str(k), "doc_ids": v} for k, v in negatives_dict.items()]
    negatives = sorted(negatives, key=lambda x: x["id"])
    write_jsonl(negatives, train_negatives_path())


def download_train_triples():
    if not train_triples_path().exists():
        dataset = ir_datasets.load("msmarco-passage/train/triples-small")
        write_list(
            tqdm(
                dataset.docpairs_iter(),
                total=dataset.docpairs_count(),
                dynamic_ncols=True,
                desc="Exporting train triples",
            ),
            path=train_triples_path(),
            callback=lambda x: f"{x.query_id}\t{x.doc_id_a}\t{x.doc_id_b}",
        )


def download_train_teacher_run():
    # Download -----------------------------------------------------------------
    url = "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
    download_file(
        url, train_teacher_run_compressed_path(), "Downloading teacher train run"
    )

    # Extract ------------------------------------------------------------------
    with gzip.open("cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz", "rb") as f:
        teacher_run = pickle.load(f)

    teacher_run = [{"id": k, "run": v} for k, v in teacher_run.items()]

    for i, x in enumerate(teacher_run):
        run = sorted(x["run"].items(), key=lambda x: x[1], reverse=True)
        del x["run"]
        x["doc_ids"] = [x[0] for x in run]
        x["scores"] = [x[1] for x in run]
        teacher_run[i] = x

    teacher_run = sorted(teacher_run, key=lambda x: x["id"])

    write_jsonl(teacher_run, train_teacher_run_path())


# def download_val():
#     if not val_qrels_path().exists() or not val_queries_path().exists():
#         dataset = ir_datasets.load("msmarco-passage/dev/small")
#         dev_q_ids = {str(q.query_id) for q in dataset.queries_iter()}

#         dataset = ir_datasets.load("msmarco-passage/dev/judged")
#         val_q_ids = [str(q.query_id) for q in dataset.queries_iter()]
#         val_q_ids = [q_id for q_id in val_q_ids if q_id not in dev_q_ids]

# qrels = defaultdict(dict)
# for q in dataset.qrels_iter():
#     qrels[str(q.query_id)][str(q.doc_id)] = q.relevance

# write_json(qrels, dev_qrels_path)
# write_jsonl(
#     dataset.queries_iter(),
#     path=dev_queries_path(),
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
        qrels_path=dev_qrels_path(),
        queries_path=dev_queries_path(),
        bm25_doc_ids_path=dev_bm25_doc_ids_path(),
    )


def download_trec_dl_2019():
    download_eval_data(
        ir_datasets_id="msmarco-passage/trec-dl-2019/judged",
        qrels_path=trec_dl_2019_qrels_path(),
        queries_path=trec_dl_2019_queries_path(),
        bm25_doc_ids_path=trec_dl_2019_bm25_doc_ids_path(),
    )


def download_trec_dl_2020():
    download_eval_data(
        ir_datasets_id="msmarco-passage/trec-dl-2020/judged",
        qrels_path=trec_dl_2020_qrels_path(),
        queries_path=trec_dl_2020_queries_path(),
        bm25_doc_ids_path=trec_dl_2020_bm25_doc_ids_path(),
    )


def download_trec_dl_hard():
    download_eval_data(
        ir_datasets_id="msmarco-passage/trec-dl-hard",
        qrels_path=trec_dl_hard_qrels_path(),
        queries_path=trec_dl_hard_queries_path(),
        bm25_doc_ids_path=trec_dl_hard_bm25_doc_ids_path(),
    )


def download_msmarco():
    download_collection()
    download_train_queries()
    download_train_relevants_and_negatives()
    download_train_triples()
    # download_val()
    download_dev()
    download_trec_dl_2019()
    download_trec_dl_2020()
    download_trec_dl_hard()
