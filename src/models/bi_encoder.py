import torch
from pytorch_lightning import LightningModule
from torch import Tensor, einsum, nn
from torch.nn.functional import normalize
from torchmetrics import Accuracy
from transformers import AutoModel

from .utils import configure_optimizers_and_schedulers


class MaskedMeanPooler(nn.Module):
    def __init__(self, eps: float = 1e-08):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        embeddings = input.last_hidden_state
        numerators = einsum("xyz,xy->xz", embeddings, mask)
        denominators = mask.sum(dim=1, keepdim=True)
        return numerators / torch.clamp(denominators, min=self.eps)


class CLS_Pooler(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.last_hidden_state[:, 0]


class Pooler(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.pooler_output


class BiEncoder(LightningModule):
    def __init__(
        self,
        encoder: str = "bert-base-uncased",
        pooling_strategy: str = "mean",
        normalize_embeddings: bool = False,
        share_weights: bool = True,
        learning_rate: float = 3e-6,
        logit_scale: float = 20.0,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        scheduler_config: dict = None,
    ):
        super().__init__()

        # Architecture ---------------------------------------------------------
        self.query_encoder = AutoModel.from_pretrained(encoder)
        self.doc_encoder = (
            self.query_encoder if share_weights else AutoModel.from_pretrained(encoder)
        )

        if pooling_strategy == "mean":
            self.pooling_layer = MaskedMeanPooler()
        elif pooling_strategy == "cls":
            self.pooling_layer = CLS_Pooler()
        elif pooling_strategy == "pooler":
            self.pooling_layer = Pooler()
        else:
            raise NotImplementedError("Invalid pooling strategy")

        self.normalize_embeddings = normalize_embeddings

        # Training -------------------------------------------------------------
        self.logit_scale = logit_scale if normalize_embeddings else 1.0

        # Loss function --------------------------------------------------------
        self.criterion = criterion

        # Optimizer ------------------------------------------------------------
        self.optimizer = torch.optim.AdamW
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config

        # Metric ---------------------------------------------------------------
        self.accuracy = Accuracy(task="binary")

        # Other ----------------------------------------------------------------
        self.pooling_strategy = pooling_strategy

    def pooling(self, output: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling_strategy == "mean":
            return self.pooling_layer(output, attention_mask)
        else:
            return self.pooling_layer(output)

    def embed_queries(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.query_encoder(input_ids, attention_mask)
        embeddings = self.pooling(output, attention_mask)

        if self.normalize_embeddings:
            embeddings = normalize(embeddings, dim=-1)

        return embeddings

    def embed_docs(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.doc_encoder(input_ids, attention_mask)
        embeddings = self.pooling(output, attention_mask)

        if self.normalize_embeddings:
            embeddings = normalize(embeddings, dim=-1)

        return embeddings

    def scoring(self, Q: Tensor, D: Tensor) -> Tensor:
        return einsum("xz,xz->x", Q, D)

    def listwise_scoring(self, Q: Tensor, D: Tensor) -> Tensor:
        n_docs_per_query = len(D) // len(Q)
        Q = Q.repeat_interleave(n_docs_per_query, dim=0)

        return self.scoring(Q, D)

    def compute_accuracy(self, pos_scores: Tensor, neg_scores: Tensor) -> float:
        return self.accuracy(
            torch.where(pos_scores > neg_scores, 1, 0),
            torch.ones(len(pos_scores), dtype=torch.long).to(self.device),
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
        Q = {k: v.to(self.device) for k, v in Q.items()}
        D = {k: v.to(self.device) for k, v in D.items()}

        Q = self.embed_queries(**Q)
        D = self.embed_docs(**D)

        scores = self.listwise_scoring(Q, D)
        scores = scores.reshape(len(Q), len(D) // len(Q))
        scores, indices = torch.topk(scores, k, dim=-1)

        return indices, scores

    def configure_optimizers(self):
        return configure_optimizers_and_schedulers(self)
