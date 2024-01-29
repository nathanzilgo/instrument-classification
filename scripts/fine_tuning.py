import json
import pandas as pd

from inda_mir.modeling.models.lgbm import LightGBMClassifier

from inda_mir.modeling.train_test_split.cv_split import CVSplitter
from sklearn.model_selection import GridSearchCV
from scripts.util.fine_tuning_params import BestModelParams

from scripts.util.config import instrument_classification_config as icc


def main_fine_tuning():
    TRAINED_FEATURES = icc['outputs']['FEATURES_TRAINED']
    FINE_TUNING_OUTPUT = icc['outputs']['GRID_PARAMS']
    GRID_OUTPUT = icc['outputs']['GRID_MODEL']
    MODEL_OUTPUT_NAME = 'lgbm_retrained_gridsearch'

    SEARCH_PARAMS = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [15, 30, 50, 75],
        'num_leaves': [59, 97, 257, 511],
        'feature_fraction': [0.8, 0.9, 0.95],
        'subsample': [0.2, 0.4],
        'max_bin': [255],
        'lambda_l1': [0, 2.5, 5],
        'lambda_l2': [0, 2.5, 5],
        'is_unbalance': [True, False],
    }

    FIXED_PARAMS = {
        'boosting': 'gbdt',
        'early_stopping_rounds': 30,
    }

    features = pd.read_csv(TRAINED_FEATURES)

    X_train = features
    y_train = features['label']

    cvl = [
        (t1, t2) for t1, t2 in CVSplitter(n_splits=4).split(X_train, y_train)
    ]

    model = LightGBMClassifier(**FIXED_PARAMS).model

    X_train = features.drop(
        ['filename', 'frame', 'track_id', 'label'], axis=1
    ).to_numpy()
    y_train = features['label'].to_numpy()

    gscv = GridSearchCV(
        estimator=model,
        param_grid=SEARCH_PARAMS,
        cv=cvl,
        scoring='accuracy',
        n_jobs=-1,  # use all CPU cores
        verbose=4,  # for score, fold, compute time logs
        return_train_score=True,  # insights on how different parameter settings impact the overfitting/underfitting trade-off
    ).fit(X_train, y_train)

    print(gscv.best_params_)
    print(gscv.best_score_)

    bp = gscv.best_params_
    bp['best_score'] = gscv.best_score_

    best_model_params = BestModelParams.validate(bp)
    json_string = best_model_params.json()

    with open(FINE_TUNING_OUTPUT, 'w') as f:
        json.dump(json_string, f, indent=4)

    model.set_params(**gscv.best_params_)

    model.save_model(
        path=GRID_OUTPUT,
        model_name=MODEL_OUTPUT_NAME,
    )


if __name__ == '__main__':
    main_fine_tuning()
