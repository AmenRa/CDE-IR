import string
from typing import Dict

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torch import Tensor, einsum, nn, tensor
from torchmetrics import Accuracy
from transformers import AutoConfig, AutoModel, AutoTokenizer


class ColBERT(LightningModule):
    def __init__(
        self,
        encoder: str = "bert-base-uncased",
        embedding_dim: int = 128,
        share_weights: bool = True,
        learning_rate: float = 3e-6,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        scheduler_config: dict = None,
    ):
        super().__init__()

        # Architecture ---------------------------------------------------------
        self.query_encoder = AutoModel.from_pretrained(encoder)
        self.query_compressor = nn.Linear(
            in_features=AutoConfig.from_pretrained(encoder).hidden_size,
            out_features=embedding_dim,
            bias=False,
        )

        if share_weights:
            self.doc_encoder = self.query_encoder
            self.doc_compressor = self.query_compressor
        else:
            self.doc_encoder = AutoModel.from_pretrained(encoder)
            self.doc_compressor = nn.Linear(
                in_features=AutoConfig.from_pretrained(encoder).hidden_size,
                out_features=embedding_dim,
                bias=False,
            )

        self.normalize = torch.nn.functional.normalize

        # Loss function --------------------------------------------------------
        self.criterion = criterion

        # Optimizer ------------------------------------------------------------
        self.optimizer = torch.optim.AdamW
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config

        # Other ----------------------------------------------------------------
        self.embedding_dim = embedding_dim  # Save embedding_dim to access it if needed

        # Metric ---------------------------------------------------------------
        self.accuracy = Accuracy(task="binary")

        # Punctuation tokens ---------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(encoder)
        # Get punctuation token ids from tokenizer
        punctuation_token_ids = tokenizer(string.punctuation, add_special_tokens=False)[
            "input_ids"
        ]
        # Add pad_token_id to punctuation_token_ids as we need to mask it too
        punctuation_token_ids.append(tokenizer.pad_token_id)
        self.punctuation_token_ids = tensor(punctuation_token_ids)

    def maxsim(self, Q_emb: Tensor, D_emb: Tensor) -> Tensor:
        """Computes pairwise maximum similarity as defined by Khattab et al., SIGIR '20 - https://dl.acm.org/doi/10.1145/3397271.3401075

        Args:
            Q_emb (Tensor): [batch_size, n_tokens, embedding_dim]
            D_emb (Tensor): [batch_size, n_tokens, embedding_dim]

        Returns:
            Tensor: [batch_size]

        """

        term_sim_scores = einsum("xbz,xdz->xbd", Q_emb, D_emb)
        max_term_sim_scores = term_sim_scores.max(dim=-1).values

        return max_term_sim_scores.sum(-1)

    def listwise_maxsim(self, Q_emb: Tensor, D_emb: Tensor) -> Tensor:
        """Computes `maxsim` for each query-document pre-defined combination.

        Args:
            Q_emb (Tensor): [batch_size, n_tokens, embedding_dim]
            D_emb (Tensor): [batch_size * n_docs_per_query, n_tokens, embedding_dim]

        Returns:
            Tensor: [batch_size * n_docs_per_query]
        """

        # Repeat query embeddings to match document embeddings
        n_docs_per_query = D_emb.shape[0] // Q_emb.shape[0]
        Q_emb = Q_emb.repeat_interleave(n_docs_per_query, dim=0)

        return self.maxsim(Q_emb, D_emb)

    def in_batch_maxsim(self, Q_emb: Tensor, D_emb: Tensor) -> Tensor:
        """Computes `maxsim` for each possible query-document combination in the batch.

        Args:
            Q_emb (Tensor): [batch_size, n_tokens, embedding_dim]
            D_emb (Tensor): [2 * batch_size, n_tokens, embedding_dim]

        Returns:
            Tensor: [2 * batch_size]
        """

        term_sim_scores = einsum("xbz,ydz->xybd", Q_emb, D_emb)
        max_term_sim_scores = term_sim_scores.max(dim=-1).values
        return max_term_sim_scores.sum(dim=-1)

    def embed_queries(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encodes queries with the defined encoder followed by a linear layer for compression and L2 normalization.

        Args:
            input_ids (Tensor): [batch_size, n_tokens]
            attention_mask (Tensor): [batch_size, n_tokens]

        Returns:
            Tensor: [batch_size, n_tokens, embedding_dim]
        """
        embeddings = self.query_encoder(input_ids, attention_mask).last_hidden_state
        embeddings = self.query_compressor(embeddings)
        return self.normalize(embeddings, dim=-1)

    def get_punctuation_mask(self, input_ids: Tensor) -> Tensor:
        """True where input is NOT a punctuation mark (or a padding token).

        Args:
            input_ids (Tensor): [batch_size, n_tokens]

        Returns:
            Tensor: [batch_size, n_tokens]
        """

        return ~(sum(input_ids == i for i in self.punctuation_token_ids).bool())

    def mask_punctuation(self, emb: Tensor, input_ids: Tensor) -> Tensor:
        """Zero out punctuation (and padding embeddings).

        Args:
            emb (Tensor): [batch_size, n_tokens, embedding_dim]
            input_ids (Tensor): [batch_size, n_tokens]

        Returns:
            Tensor: [batch_size, n_tokens, embedding_dim]
        """

        punctuation_mask = self.get_punctuation_mask(input_ids)
        punctuation_mask = punctuation_mask.unsqueeze(-1).expand(emb.size())

        return emb.masked_fill(punctuation_mask == 0, 0.0)

    def embed_docs(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encodes documents with the defined encoder followed by a linear layer for compression and L2 normalization.

        Args:
            input_ids (Tensor): [batch_size, n_tokens]
            attention_mask (Tensor): [batch_size, n_tokens]

        Returns:
            Tensor: [batch_size, n_tokens, embedding_dim]
        """
        embeddings = self.doc_encoder(input_ids, attention_mask).last_hidden_state
        embeddings = self.doc_compressor(embeddings)
        embeddings = self.normalize(embeddings, dim=-1)

        return self.mask_punctuation(embeddings, input_ids)

    # In PyTorch Lightning, forward defines inference
    def forward(self, Q: Dict[str, Tensor], D: Dict[str, Tensor], k: int) -> Tensor:
        Q = {k: v.to(self.device) for k, v in Q.items()}
        D = {k: v.to(self.device) for k, v in D.items()}

        Q = self.embed_queries(**Q)
        D = self.embed_docs(**D)

        scores = self.listwise_maxsim(Q, D)
        scores = scores.reshape(len(Q), len(D) // len(Q))
        scores, indices = torch.sort(scores, dim=-1, descending=True, stable=True)

        return indices[:, :k], scores[:, :k]

    def compute_accuracy(self, pos_scores: Tensor, neg_scores: Tensor) -> float:
        return self.accuracy(
            torch.where(pos_scores > neg_scores, 1, 0),
            torch.ones(len(pos_scores), dtype=torch.long).to(self.device),
        )

    # In PyTorch Lightning, training_step is called during training.
    # It is used to separate the training forward from the inference forward.
    def training_step(
        self,
        batch,
        batch_idx,
    ) -> float:
        """Training step modeled around STAR training strategy proposed by Zhan et al., SIGIR '21 - https://dl.acm.org/doi/10.1145/3404835.3462880.

        Args:
            batch: tuple containing:
                Q (Dict[str, Tensor]): Dictionary in the form of {"input_ids": [...], "attention_mask": [...]} containing the queries tokenization data. It is the same format of the HuggingFace Tokenizers.

                D (Dict[str, Tensor]): Dictionary in the form of {"input_ids": [...], "attention_mask": [...]} containing the documents tokenization data. It is the same format of the HuggingFace Tokenizers. The first half is composed of the positive documents, the second half is composed of the hard negative documents.

        Returns:
            float: Training loss.
        """
        Q, D = batch
        batch_size = len(Q["input_ids"])

        # Compute embeddings ---------------------------------------------------
        Q_emb = self.embed_queries(**Q)
        # D_emb = self.embed_docs(**D)
        # D_pos_emb, D_neg_emb = D_emb.tensor_split(2)

        # Compute scores -------------------------------------------------------
        scores = torch.zeros(batch_size, 2 * batch_size, device=self.device)
        reduction_factor = 2
        mini_batch_size = batch_size // reduction_factor
        for i in range(2 * reduction_factor):
            start, stop = i * mini_batch_size, (i + 1) * mini_batch_size
            D_emb = self.embed_docs(**{k: v[start:stop] for k, v in D.items()})
            scores[:, start:stop] = self.in_batch_maxsim(Q_emb, D_emb)
            del D_emb  # Release memory

        # scores = self.in_batch_maxsim(Q_emb, D_emb)

        # pos_scores = self.maxsim(Q_emb, D_pos_emb)
        # neg_scores = self.maxsim(Q_emb, D_neg_emb)

        # Compute loss ---------------------------------------------------------
        loss = self.criterion(scores, torch.arange(batch_size).to(self.device))

        # loss = self.criterion(
        #     torch.column_stack([pos_scores, neg_scores]),
        #     torch.zeros(len(pos_scores), dtype=torch.long, device=self.device),
        # )

        # Compute metrics ------------------------------------------------------
        pos_scores = torch.diagonal(scores[:, :batch_size], 0)
        neg_scores = torch.diagonal(scores[:, batch_size:], 0)
        accuracy = self.compute_accuracy(pos_scores, neg_scores)

        # accuracy = self.accuracy(
        #     torch.where(pos_scores > neg_scores, 1, 0),
        #     torch.ones(len(pos_scores), dtype=torch.long, device=self.device),
        # )

        # Logging --------------------------------------------------------------
        self.log("accuracy", accuracy, on_step=True, on_epoch=False)
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def forward_precomputed(self, Q_emb: Tensor, D_emb: Tensor, k: int):
        scores = self.listwise_maxsim(Q_emb, D_emb)
        scores = scores.reshape(Q_emb.shape[0], int(D_emb.shape[0] / Q_emb.shape[0]))

        scores, indices = torch.sort(
            scores,
            dim=-1,
            descending=True,
            stable=True,
        )

        return indices[:k], scores[:k]

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
