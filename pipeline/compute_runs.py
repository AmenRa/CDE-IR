import os
from math import ceil

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from ranx import Qrels, Run, evaluate
from tqdm import trange
from unified_io import join_path, read_jsonl, read_numpy, write_json

from src.utils import setup_logger


def compute_run(query_embs, doc_embs, query_index, doc_ids_map):
    batch_size = 128

    run = {}

    for i in trange(ceil(len(query_embs) / batch_size)):
        with torch.cuda.amp.autocast():
            start = i * batch_size
            stop = (i + 1) * batch_size

            scores = torch.einsum("xz,yz->yx", doc_embs, query_embs[start:stop])
            indices = torch.topk(
                scores, k=100, dim=-1, largest=True, sorted=True
            ).indices

            for j in range(len(scores)):
                _indices = indices[j].tolist()
                _scores = scores[j][indices[j]].tolist()
                doc_ids = [doc_ids_map[idx] for idx in _indices]

                query = query_index[start + j]

                run[query["id"]] = dict(zip(doc_ids, _scores))

    return run


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger,
        dir=join_path(cfg.paths.logs, cfg.model.name),
        filename="compute_runs.log",
    )

    doc_ids_map = read_jsonl(
        "datasets/msmarco_passage/collection.jsonl", callback=lambda x: x["id"]
    )
    doc_ids_map = dict(enumerate(doc_ids_map))

    doc_embs = torch.from_numpy(
        read_numpy(join_path(cfg.general.data_dir, "doc_embs.npy"))
    ).cuda()

    # Compute runs =============================================================
    for split in [
        "dev",
        "trec_dl_2019",
        "trec_dl_2020",
        # "trec_dl_hard",
    ]:
        # I/O Paths ------------------------------------------------------------
        os.makedirs(cfg.paths.runs, exist_ok=True)
        out_path = join_path(cfg.paths.runs, f"{cfg.model.name}_{split}_run.json")

        query_index = read_jsonl(f"datasets/msmarco_passage/{split}/queries.jsonl")
        query_embs = torch.from_numpy(
            read_numpy(join_path(cfg.general.data_dir, f"{split}_query_embs.npy"))
        ).cuda()

        run = compute_run(query_embs, doc_embs, query_index, doc_ids_map)
        write_json(run, out_path)

        qrels = Qrels.from_file(f"datasets/msmarco_passage/{split}/qrels.json")
        run = Run(run)

        ndcg_score = round(evaluate(qrels, run, "ndcg@10") * 100, 2)

        if split == "dev":
            mrr_score = round(evaluate(qrels, run, "mrr@10") * 100, 2)
            map_score = round(evaluate(qrels, run, "map") * 100, 2)
        else:
            mrr_score = round(evaluate(qrels, run, "mrr@10-l2") * 100, 2)
            map_score = round(evaluate(qrels, run, "map-l2") * 100, 2)

        logger.info(f"MRR@10: {mrr_score}\tNDCG@10: {ndcg_score}\tMAP: {map_score}")


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
