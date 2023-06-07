from math import ceil

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torch import Tensor, column_stack, long, nn, zeros
from torchmetrics import Accuracy
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
)


class CrossEncoder(LightningModule):
    def __init__(
        self,
        language_model: str = "bert-base-uncased",
        learning_rate: float = 3e-6,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        scheduler_config: dict = None,
    ):
        super().__init__()

        # Architecture ---------------------------------------------------------
        if "distilbert" in language_model:
            self.encoder = DistilBertForSequenceClassification.from_pretrained(
                language_model
            )
        else:
            self.encoder = BertForSequenceClassification.from_pretrained(language_model)

        # Loss function --------------------------------------------------------
        self.criterion = criterion

        # Optimizer ------------------------------------------------------------
        self.optimizer = torch.optim.AdamW
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config

        # Metric ---------------------------------------------------------------
        self.accuracy = Accuracy(task="binary")

    def compute_loss(self, pos_scores: Tensor, neg_scores: Tensor) -> float:
        return self.criterion(
            column_stack([pos_scores, neg_scores]),
            zeros(len(pos_scores), dtype=torch.long, device=self.device),
        )

    def compute_accuracy(self, pos_scores: Tensor, neg_scores: Tensor) -> float:
        return self.accuracy(
            torch.where(pos_scores > neg_scores, 1, 0),
            torch.ones(len(pos_scores), dtype=torch.long).to(self.device),
        )

    def training_step(self, batch, batch_idx) -> float:
        pos_tokens, neg_tokens = batch
        batch_size = len(pos_tokens["input_ids"])

        # Compute scores -------------------------------------------------------
        pos_scores = self.encoder(**pos_tokens).logits[:, 0]
        neg_scores = self.encoder(**neg_tokens).logits[:, 0]
        scores = column_stack([pos_scores, neg_scores])

        # Compute loss ---------------------------------------------------------
        loss = self.criterion(scores, zeros(batch_size, dtype=long).to(self.device))

        # Compute metrics ------------------------------------------------------
        accuracy = self.compute_accuracy(pos_scores, neg_scores)

        # Logging --------------------------------------------------------------
        self.log("accuracy", accuracy, on_step=True, on_epoch=False)
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def forward(self, batch: list[dict[str, Tensor]], k: int):
        indices, scores = [], []

        for tokens in batch:
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            _scores = torch.zeros((len(tokens["input_ids"])))
            for i in range(ceil(len(tokens["input_ids"]) // 500)):
                start, stop = i * 500, (i + 1) * 500
                _tokens = {k: v[start:stop] for k, v in tokens.items()}
                _scores[start:stop] = self.encoder(**_tokens).logits[:, 0]
            _scores, _indices = torch.topk(_scores, k, dim=-1)

            indices.append(_indices)
            scores.append(_scores)

        return indices, scores

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)

        if self.scheduler_config is None:
            return optimizer

        self.scheduler_config["optimizer"] = optimizer
        scheduler = instantiate(self.scheduler_config)

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                }
            ],
        )
