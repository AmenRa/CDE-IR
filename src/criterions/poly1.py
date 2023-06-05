from torch import Tensor, nn
from torch.nn.functional import one_hot, softmax


class Poly1(nn.Module):
    def __init__(
        self,
        eps: float = 2.0,
        weight: Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.weight = weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def compute_pt(self, input: Tensor, target: Tensor) -> Tensor:
        pt = one_hot(target, input.size(1)) * softmax(input, 1)
        return pt.sum(dim=-1)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = self.cross_entropy(input, target)
        pt = self.compute_pt(input, target)
        loss = ce_loss + self.eps * (1 - pt)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
