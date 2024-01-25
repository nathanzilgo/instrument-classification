from pydantic import BaseModel


class BestModelParams(BaseModel):
    learning_rate: float
    max_depth: int
    num_leaves: int
    feature_fraction: float
    subsample: float
    max_bin: int
    lambda_l1: float
    lambda_l2: float
    is_unbalance: bool
    best_score: float
