import torch
from torch.utils.data import Dataset

from models import Metrics, RecommendResult

def evaluate(model: torch.nn.Module, top_k: int = 10, device: str = "gpu") -> Metrics:
    model.eval()

    with torch.no_grad():
        pass