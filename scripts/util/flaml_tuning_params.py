from pydantic import BaseModel


class FlamlTuningParams(BaseModel):
    n_estimators: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    log_max_bin: int
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    best_loss: float
