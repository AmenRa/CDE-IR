import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.functional import one_hot, softmax


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 1.0, num_classes: int = 2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # CCE
        ce = self.cross_entropy(input, target)

        # RCE
        input = softmax(input, dim=1)
        input = torch.clamp(input, min=1e-7, max=1.0)
        label_one_hot = one_hot(target, input.shape[-1]).float().to(input.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        rce = -1 * torch.sum(input * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
