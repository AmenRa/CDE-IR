import os

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from oneliner_utils import join_path
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from src.collators import CrossTrainCollator, TrainCollator
from src.datasets.msmarco_passage import TrainDataset, download_msmarco
from src.utils import get_pl_callbacks, get_pl_loggers, setup_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger,
        dir=join_path(cfg.general.logs_dir, cfg.model.name),
        filename="training.log",
    )

    # Set seeds for reproducibility --------------------------------------------
    logger.info("Random seeds")
    seed_everything(cfg.general.seed)

    # Download & prepare dataset -----------------------------------------------
    logger.info("Downloading & Preparing Dataset")
    download_msmarco()

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
        train_collator = CrossTrainCollator(cross_tokenizer)
    else:
        train_collator = TrainCollator(query_tokenizer, doc_tokenizer)

    # Dataset  -----------------------------------------------------------------
    logger.info("Dataset")
    n_samples = cfg.dataloader.batch_size * cfg.trainer.limit_train_batches
    train_dataset = TrainDataset(n_samples)

    # Dataloader  --------------------------------------------------------------
    logger.info("Dataloader")
    train_loader = DataLoader(
        train_dataset, **cfg.dataloader, collate_fn=train_collator
    )

    # Trainer ------------------------------------------------------------------
    logger.info("Trainer")
    trainer = Trainer(
        logger=get_pl_loggers(cfg),
        callbacks=get_pl_callbacks(cfg),
        **cfg.trainer,
    )

    # Model --------------------------------------------------------------------
    logger.info("Model")
    model = instantiate(cfg.model.init, criterion=instantiate(cfg.criterion.config))

    # Scheduler ----------------------------------------------------------------
    logger.info("Scheduler")
    scheduler_config = OmegaConf.to_container(cfg.scheduler.config, resolve=True)
    steps = cfg.trainer.limit_train_batches // cfg.trainer.accumulate_grad_batches
    scheduler_config["first_cycle_steps"] = steps
    scheduler_config["warmup_steps"] = 0.1 * steps
    model.scheduler_config = scheduler_config

    # Training -----------------------------------------------------------------
    logger.info("Training")
    trainer.fit(model, train_dataloaders=train_loader)

    # Save trained model -------------------------------------------------------
    logger.info("Save model")
    trainer.save_checkpoint(join_path(cfg.general.model_dir, "model.ckpt"))

    logger.success("Training complete!")


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
