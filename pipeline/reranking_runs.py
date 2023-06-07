import os
from collections import defaultdict

import hydra
import torch
from hydra.utils import call, instantiate
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from ranx import Qrels, Run, evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
from unified_io import join_path, write_json

from src.collators import CrossEvalCollator, EvalCollator
from src.datasets.msmarco_passage import EvalDataset, get_qrels_path
from src.utils import setup_logger


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger,
        dir=join_path(cfg.paths.logs, cfg.model.name),
        filename="compute_reranking_runs.log",
    )

    # Set seeds for reproducibility --------------------------------------------
    logger.info("Random seeds")
    seed_everything(cfg.general.seed)

    # Tokenizers ---------------------------------------------------------------
    logger.info("Tokenizers")
    if "cross_encoder" in cfg.model.name:
        cross_tokenizer = instantiate(cfg.tokenizer.init)
    else:
        query_tokenizer = instantiate(cfg.tokenizer.query_tokenizer.init)
        doc_tokenizer = instantiate(cfg.tokenizer.doc_tokenizer.init)

    # Collator -----------------------------------------------------------------
    logger.info("Collator")
    if "cross_encoder" in cfg.model.name:
        eval_collator = CrossEvalCollator(cross_tokenizer)
    else:
        eval_collator = EvalCollator(query_tokenizer, doc_tokenizer)

    # Compute runs =============================================================
    for split in [
        # "dev",
        "trec-dl-2019",
        "trec-dl-2020",
    ]:
        # I/O Paths ------------------------------------------------------------
        os.makedirs(cfg.paths.runs, exist_ok=True)
        out_path = join_path(cfg.paths.runs, f"{split}_run.json")

        # logger.info("Dataset")
        dataset = EvalDataset(split=split)

        # logger.info("Dataloader")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=eval_collator,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

        # Load model -----------------------------------------------------------
        model = call(
            cfg.model.checkpoint,
            checkpoint_path=join_path(cfg.paths.model, "model.ckpt"),
        )

        model = model.cuda()
        model.eval()

        run = defaultdict(dict)

        logger.info(f"Computing {split} run")
        with torch.cuda.amp.autocast(), torch.no_grad():
            for batch in tqdm(dataloader, dynamic_ncols=True):
                if "cross_encoder" in cfg.model.name:
                    batch_query_id, batch_doc_ids, batch_tokens = batch
                    indices, sorted_scores = model.forward(batch_tokens, k=1000)
                else:
                    batch_query_id, batch_doc_ids, batch_query, batch_docs = batch
                    indices, sorted_scores = model.forward(
                        batch_query, batch_docs, k=1000
                    )

                # Update run
                for i, (q_id, doc_ids) in enumerate(zip(batch_query_id, batch_doc_ids)):
                    for j, idx in enumerate(indices[i]):
                        try:
                            doc_id = doc_ids[idx]
                            if doc_id != "fake_doc":
                                run[q_id][doc_id] = sorted_scores[i][j].item()
                        except:
                            pass

        run = Run(run)
        run.save(out_path)

        qrels = Qrels.from_file(get_qrels_path(split))

        if split == "dev":
            mrr_score = round(evaluate(qrels, run, "mrr@10") * 100, 2)
            map_score = round(evaluate(qrels, run, "map") * 100, 2)
            logger.info(f"MRR@10: {mrr_score}\MAP: {map_score}")
        else:
            ndcg_score = round(evaluate(qrels, run, "ndcg@10") * 100, 2)
            map_score = round(evaluate(qrels, run, "map-l2") * 100, 2)
            logger.info(f"NDCG@10: {ndcg_score}\tMAP: {map_score}")


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
