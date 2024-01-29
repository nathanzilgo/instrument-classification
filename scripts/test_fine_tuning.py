import pandas as pd
from unittest import mock

from scripts.fine_tuning import main_fine_tuning


@mock.patch('json.dump')
@mock.patch('scripts.fine_tuning.BestModelParams')
@mock.patch('scripts.fine_tuning.GridSearchCV')
@mock.patch('scripts.fine_tuning.LightGBMClassifier')
@mock.patch('scripts.fine_tuning.CVSplitter')
@mock.patch('scripts.fine_tuning.pd.read_csv')
def test_script_execution(
    mock_read_csv: mock.Mock,
    mock_splitter: mock.Mock,
    mock_model: mock.Mock,
    mock_gscv: mock.Mock,
    mock_ft_params: mock.Mock,
    mock_json: mock.Mock,
) -> None:
    model_return_value = mock_model.return_value.model
    validate_call = mock_ft_params.validate

    main_fine_tuning()

    mock_splitter.return_value.split.assert_called_once()
    mock_gscv.return_value.fit.assert_called_once()

    model_return_value.set_params.assert_called_once()
    model_return_value.save_model.assert_called_once_with(
        path='./output/models/gbm_retrained_finetuned_gridsearch.pkl',
        model_name='lgbm_retrained_gridsearch',
    )

    validate_call.assert_called_once()
    validate_call.return_value.json.assert_called_once()

    mock_json.assert_called_once()
