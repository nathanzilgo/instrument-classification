import json
from flaml import AutoML

import pandas as pd

from scripts.util.config import instrument_classification_config as icc
from scripts.util.flaml_tuning_params import FlamlTuningParams

TRAINED_FEATURES = icc['outputs']['FEATURES_TRAINED']
VALIDATION_FEATURES = icc['outputs']['FEATURES_VALIDATION']
FINE_TUNING_OUTPUT = icc['outputs']['FLAML_PARAMS']

features = pd.read_csv(TRAINED_FEATURES)
validation_features = pd.read_csv(VALIDATION_FEATURES)

X_train = features.drop(
    ['filename', 'frame', 'track_id', 'label'], axis=1
).to_numpy()
y_train = features['label'].to_numpy()

X_val = validation_features.drop(
    ['filename', 'frame', 'track_id', 'label'], axis=1
).to_numpy()
y_val = validation_features['label'].to_numpy()

automl = AutoML()

# Set max number of iterations here: (around 200 suits well)
MAX_ITER = 20
automl_settings = {
    'max_iter': MAX_ITER,  # setting the time budget
    'metric': 'macro_f1',
    'task': 'classification',
    'estimator_list': ['lgbm'],
    'log_file_name': 'lgbm_flaml_tune.log',  # set the file to save the log for HPO
    'log_type': 'all',  # the log type for trials: "all" if logging all the trials, "better" if only keeping the better trials
    'keep_search_state': True,  # keeping the search state
}

automl.fit(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    seed=42,
    **automl_settings
)

bc = automl.best_config
bc['best_loss'] = automl.best_loss

print('Best hyperparmeter config:', automl.best_config)
print('Best auc on validation data: {0:.4g}'.format(1 - automl.best_loss))
print(
    'Training duration of best run: {0:.4g} s'.format(
        automl.best_config_train_time
    )
)
print(automl.model.estimator)

flaml_tuning_params = FlamlTuningParams.validate(bc)
json_string = flaml_tuning_params.json()

with open(FINE_TUNING_OUTPUT, 'w') as f:
    json.dump(json_string, f, indent=4)
