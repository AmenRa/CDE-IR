import os
from collections import defaultdict

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from oneliner_utils import join_path, write_json
from pytorch_lightning import seed_everything
from ranx import Qrels, Run, evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.collators import EvalCollator
from src.datasets.msmarco_passage import EvalDataset
from src.models import BiEncoder
from src.tokenizers import DocTokenizer, QueryTokenizer
from src.utils import setup_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger,
        dir=join_path(cfg.general.logs_dir, cfg.model.name),
        filename="compute_reranking_runs.log",
    )

    # Set seeds for reproducibility --------------------------------------------
    logger.info("Random seeds")
    seed_everything(cfg.general.seed)

    # Tokenizers ---------------------------------------------------------------
    logger.info("Tokenizers")
    query_tokenizer = QueryTokenizer(**cfg.tokenizer.query_tokenizer.init)
    doc_tokenizer = DocTokenizer(**cfg.tokenizer.doc_tokenizer.init)

    # Collator -----------------------------------------------------------------
    logger.info("Collator")
    eval_collator = EvalCollator(
        query_tokenizer=query_tokenizer, doc_tokenizer=doc_tokenizer
    )

    # Compute runs =============================================================
    for split in [
        # "dev",
        "trec_dl_2019",
        "trec_dl_2020",
    ]:
        # I/O Paths ------------------------------------------------------------
        os.makedirs(cfg.general.runs_dir, exist_ok=True)
        out_path = join_path(cfg.general.runs_dir, f"{cfg.model.name}_{split}_run.json")

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
        model = BiEncoder.load_from_checkpoint(
            join_path(cfg.general.model_dir, "model.ckpt"),
            **cfg.model.params,
        ).cuda()
        model.eval()

        run = defaultdict(dict)

        logger.info(f"Computing {split} run")
        with torch.cuda.amp.autocast(), torch.no_grad():
            for batch in tqdm(dataloader, dynamic_ncols=True):
                (
                    batch_query_id,
                    batch_rel_doc_ids,
                    batch_doc_ids,
                    batch_query,
                    batch_docs,
                ) = batch

                batch_query["input_ids"] = batch_query["input_ids"].cuda()
                batch_query["attention_mask"] = batch_query["attention_mask"].cuda()
                batch_docs["input_ids"] = batch_docs["input_ids"].cuda()
                batch_docs["attention_mask"] = batch_docs["attention_mask"].cuda()

                indices, sorted_scores = model.forward(batch_query, batch_docs, k=1000)

                # Update run
                for i, (q_id, doc_ids) in enumerate(zip(batch_query_id, batch_doc_ids)):
                    for j, idx in enumerate(indices[i]):
                        try:
                            doc_id = doc_ids[idx]
                            if doc_id != "fake_doc":
                                run[q_id][doc_id] = sorted_scores[i][j].item()
                        except:
                            pass

        write_json(run, out_path)

        qrels = Qrels.from_file(f"datasets/msmarco_passage/{split}/qrels.json")
        run = Run(run)

        # for q_id in list(qrels.qrels):
        #     if q_id not in list(run.run):
        #         del qrels.qrels[q_id]

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
