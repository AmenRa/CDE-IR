import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torch import Tensor, einsum, nn
from torch.nn.functional import normalize
from torchmetrics import Accuracy
from transformers import AutoConfig, AutoModel


class MaskedMeanPooling(nn.Module):
    def __init__(self, eps: float = 1e-08):
        super().__init__()
        self.eps = eps

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        numerators = einsum("xyz,xy->xyz", embeddings, mask).sum(dim=1)
        denominators = mask.sum(dim=1, keepdim=True)
        return numerators / torch.clamp(denominators, min=self.eps)


class BiEncoder(LightningModule):
    def __init__(
        self,
        language_model: str = "bert-base-uncased",
        position_embedding_type: str = "absolute",
        normalize_embeddings: bool = True,
        learning_rate: float = 3e-6,
        logit_scale: float = 20.0,
        scheduler_config: dict = None,
    ):
        super().__init__()

        # Architecture ---------------------------------------------------------
        cfg = AutoConfig.from_pretrained(language_model)
        cfg.position_embedding_type = position_embedding_type
        self.language_model = AutoModel.from_pretrained(language_model, config=cfg)
        self.pooling_layer = MaskedMeanPooling()
        self.normalize_embeddings = normalize_embeddings

        # Training -------------------------------------------------------------
        self.logit_scale = logit_scale if normalize_embeddings else 1.0

        # Loss function --------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer ------------------------------------------------------------
        self.optimizer = torch.optim.AdamW
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config

        # Metric ---------------------------------------------------------------
        self.accuracy = Accuracy(task="binary")

    def embed_queries(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        embeddings = self.language_model(input_ids, attention_mask).last_hidden_state
        embeddings = self.pooling_layer(embeddings, attention_mask)

        if self.normalize_embeddings:
            embeddings = normalize(embeddings, dim=-1)

        return embeddings

    def embed_docs(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.embed_queries(input_ids, attention_mask)

    def scoring(self, Q: Tensor, D: Tensor) -> Tensor:
        return einsum("xz,xz->x", Q, D)

    def listwise_scoring(self, Q: Tensor, D: Tensor) -> Tensor:
        n_docs_per_query = len(D) // len(Q)
        Q = Q.repeat_interleave(n_docs_per_query, dim=0)

        return self.scoring(Q, D)

    def compute_accuracy(self, pos_scores: Tensor, neg_scores: Tensor) -> float:
        return self.accuracy(
            torch.where(pos_scores > neg_scores, 1, 0),
            torch.ones(len(pos_scores), dtype=torch.long, device=self.device),
        )

    def training_step(self, batch, batch_idx) -> float:
        # Unpack batch ---------------------------------------------------------
        Q, D = batch
        batch_size = len(Q["input_ids"])

        # Compute embeddings ---------------------------------------------------
        Q_emb = self.embed_queries(**Q)
        D_emb = self.embed_docs(**D)

        # Compute scores -------------------------------------------------------
        scores = torch.mm(Q_emb, D_emb.T) * self.logit_scale

        # Compute loss ---------------------------------------------------------
        loss = self.criterion(scores, torch.arange(batch_size).to(self.device))

        # Compute metrics ------------------------------------------------------
        pos_scores = torch.diagonal(scores[:, :batch_size], 0)
        neg_scores = torch.diagonal(scores[:, batch_size:], 0)
        accuracy = self.compute_accuracy(pos_scores, neg_scores)

        # Logging --------------------------------------------------------------
        self.log("accuracy", accuracy, on_step=True, on_epoch=False)
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def forward(self, Q: dict[str, Tensor], D: dict[str, Tensor], k: int):
        Q = self.embed_queries(**Q)
        D = self.embed_docs(**D)

        scores = self.listwise_scoring(Q, D)
        scores = scores.reshape(len(Q), len(D) // len(Q))
        scores, indices = torch.sort(scores, dim=-1, descending=True, stable=True)

        return indices[:, :k], scores[:, :k]

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
