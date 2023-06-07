from math import ceil

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, column_stack, long, nn, zeros
from torchmetrics import Accuracy
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
)

from .utils import configure_optimizers_and_schedulers


class CrossEncoder(LightningModule):
    def __init__(
        self,
        encoder: str = "bert-base-uncased",
        learning_rate: float = 3e-6,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        scheduler_config: dict = None,
    ):
        super().__init__()

        # Architecture ---------------------------------------------------------
        if "distilbert" in encoder:
            self.encoder = DistilBertForSequenceClassification.from_pretrained(encoder)
        else:
            self.encoder = BertForSequenceClassification.from_pretrained(encoder)

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
            # Move input to device ---------------------------------------------
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            # Compute scores ---------------------------------------------------
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
        return configure_optimizers_and_schedulers(self)
