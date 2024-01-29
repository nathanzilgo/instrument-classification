from unittest import mock

import pandas as pd

from scripts.flaml_fine_tuning import main_flaml_fine_tuning


@mock.patch('json.dump')
@mock.patch('scripts.flaml_fine_tuning.FlamlTuningParams')
@mock.patch('scripts.flaml_fine_tuning.AutoML')
@mock.patch('scripts.flaml_fine_tuning.pd.read_csv')
def test_script_execution(
    mock_read_csv: mock.Mock,
    auto_ml_mock: mock.Mock,
    flaml_params_model_mock: mock.Mock,
    json_mock: mock.Mock,
) -> None:
    auto_ml = auto_ml_mock.return_value
    auto_ml.best_config = {
        'n_estimators': 4,
        'num_leaves': 10,
        'min_child_samples': 19,
        'learning_rate': 0.16620632646655242,
        'log_max_bin': 9,
        'colsample_bytree': 0.8864313721841287,
        'reg_alpha': 0.0021126916465507638,
        'reg_lambda': 4.381679244171673,
    }
    auto_ml.best_loss = 0.2078887436719623
    auto_ml.best_config_train_time = 0.2341234

    expected_model_contructor = auto_ml.best_config | {
        'best_loss': auto_ml.best_loss
    }

    main_flaml_fine_tuning()

    mock_read_csv.assert_has_calls(
        [
            mock.call(
                './output/features/our_extractor_features_with_brass.csv'
            ),
            mock.call('./output/features/validation_features.csv'),
        ]
    )
    auto_ml.fit.assert_called_once()
    flaml_params_model_mock.validate.assert_called_once_with(
        expected_model_contructor
    )
    json_mock.assert_called_once()
