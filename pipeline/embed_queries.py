import json
import os
from math import ceil
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from oneliner_utils import join_path, read_jsonl, write_numpy
from pytorch_lightning import seed_everything
from tqdm import tqdm

from src.models import BiEncoder
from src.tokenizers import QueryTokenizer
from src.utils import setup_logger

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger,
        dir=join_path(cfg.general.logs_dir, cfg.model.name),
        filename="embed_queries.log",
    )

    # Set seeds for reproducibility --------------------------------------------
    logger.info("Random seeds")
    seed_everything(cfg.general.seed)

    # Load model ---------------------------------------------------------------
    logger.info("Model")
    model = BiEncoder.load_from_checkpoint(
        join_path(cfg.general.model_dir, "model.ckpt"), **cfg.model.params
    ).cuda()
    model.eval()

    # Load tokenizer -----------------------------------------------------------
    logger.info("Tokenizer")
    query_tokenizer = QueryTokenizer(**cfg.tokenizer.query_tokenizer.init)

    for split in ["dev", "trec_dl_2019", "trec_dl_2020"]:
        # I/O Paths ------------------------------------------------------------
        # Input
        queries_path = f"datasets/msmarco_passage/{split}/queries.jsonl"

        # Output
        os.makedirs(cfg.general.data_dir, exist_ok=True)
        embeddings_path = Path(cfg.general.data_dir) / f"{split}_query_embs.npy"

        # Load queries ---------------------------------------------------------
        logger.info("Queries")
        queries = read_jsonl(queries_path, callback=lambda x: x["text"])

        # Compute embeddings ---------------------------------------------------
        logger.info("Embed queries")
        embeddings = []

        pbar = tqdm(
            total=len(queries),
            desc="Generating embeddings",
            position=0,
            dynamic_ncols=True,
            mininterval=1.0,
        )

        batch_size = 512
        with torch.cuda.amp.autocast(), torch.no_grad():
            for i in range(ceil(len(queries) / batch_size)):
                start, stop = i * batch_size, (i + 1) * batch_size
                input = query_tokenizer(queries[start:stop])
                input = {k: v.cuda() for k, v in input.items()}
                embeddings.append(model.embed_queries(**input))

                pbar.update(stop - start)
        pbar.close()

        embeddings = torch.cat(embeddings)
        embeddings = embeddings.detach().cpu().numpy()

        write_numpy(embeddings, embeddings_path)


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
