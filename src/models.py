import dataclasses
import torch
import pandas as pd
from typing import Dict, List


@dataclasses.dataclass(frozen=True)
class Metrics:
    hr_at_k: float
    ndcg_at_k: float
    recall_at_k: float

    def __repr__(self):
        return f"NDCG@K={self.hr_at_k:.3f}, Recall@K={self.recall_at_k:.3f}, HR@K={self.hr_at_k:.3f}"


@dataclasses.dataclass(frozen=True)
class RecommendResult:
    rating: pd.DataFrame
    # キーはユーザーID、値は推薦されたアイテムIDのリスト
    user2items: Dict[int, List[int]]


@dataclasses.dataclass(frozen=True)
class DataContainer:
    # trainはtorch.utils.data.Datasetのインスタンス
    train: torch.utils.data.Dataset
    valid: pd.DataFrame
    test: pd.DataFrame
    # ランキング指標のテストデータセット。キーはユーザーID、値はユーザーが高評価したアイテムIDのリスト
    valid_user2items: Dict[int, List[int]]
    test_user2items: Dict[int, List[int]]
    # アイテムのコンテンツ情報
    item_content: pd.DataFrame
