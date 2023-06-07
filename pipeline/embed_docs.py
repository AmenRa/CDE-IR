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
from pytorch_lightning import seed_everything
from tqdm import tqdm
from unified_io import join_path, read_jsonl, write_numpy

from src.models import BiEncoder
from src.tokenizers import DocTokenizer
from src.utils import setup_logger


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger,
        dir=join_path(cfg.paths.logs, cfg.model.name),
        filename="embed_docs.log",
    )

    # Set seeds for reproducibility --------------------------------------------
    logger.info("Random seeds")
    seed_everything(cfg.general.seed)

    # I/O Paths ----------------------------------------------------------------
    # Input
    documents_path = "datasets/msmarco_passage/collection.jsonl"

    # Output
    os.makedirs(cfg.general.data_dir, exist_ok=True)
    embeddings_path = Path(cfg.general.data_dir) / "doc_embs.npy"

    # Load model ---------------------------------------------------------------
    logger.info("Model")
    model = BiEncoder.load_from_checkpoint(
        join_path(cfg.paths.model, "model.ckpt"), **cfg.model.params
    ).cuda()
    model.eval()

    # Load tokenizer -----------------------------------------------------------
    logger.info("Tokenizer")
    doc_tokenizer = DocTokenizer(**cfg.tokenizer.doc_tokenizer.init)

    # Load docs ----------------------------------------------------------------
    logger.info("Documents")
    docs = read_jsonl(documents_path, callback=lambda x: x["text"])

    # Compute embeddings =======================================================
    logger.info("Embed documents")
    embeddings = []

    pbar = tqdm(
        total=len(docs),
        desc="Generating embeddings",
        position=0,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    batch_size = 512
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in range(ceil(len(docs) / batch_size)):
            start, stop = i * batch_size, (i + 1) * batch_size
            input = doc_tokenizer(docs[start:stop])
            input = {k: v.cuda() for k, v in input.items()}
            embeddings.append(model.embed_docs(**input))

            pbar.update(stop - start)
    pbar.close()

    embeddings = torch.cat(embeddings)
    embeddings = embeddings.detach().cpu().numpy()

    write_numpy(embeddings, embeddings_path)


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
