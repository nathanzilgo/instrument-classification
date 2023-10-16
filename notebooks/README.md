# Notebooks Detailing

|           Notebook      |                            Description                          |
|-----------------------|---------------------------------------------------------------|
| [training_testing_models.ipynb](./training_testing_models.ipynb)  | Trains a LightGBM model over a given data partition and evaluates it a sample level. It will also save the trained models in `models` directory. |
| [training_testing_models_kfold.ipynb](./training_testing_models_kfold.ipynb)  | Score ML models over a k-fold data partition. |
| [training_testing_models_w_others_threshold_analysis.ipynb](./training_testing_models_w_others_threshold_analysis.ipynb)  | Includes an analysis about the variation of the other-label threshold for the LightGBM model. |
| [training_testing_models_w_brass.ipynb](./training_testing_models_w_brass.ipynb)  | Trains a LightGBM model over a data partition including a new class brass. |
| [track_level_evaluation.ipynb](./track_level_evaluation.ipynb)  | Trains a LightGBM model over a given data partition and evaluates it a track level. |
| [fimportance_models.ipynb](./fimportance_models.ipynb)  | Includes an analysis about the feature importance of the models (RF, XGBoost and LightGBM) and evaluates then using only top-k most important features. |
| [explaining_the_models.ipynb](./explaining_the_models.ipynb)  | Includes an analysis using the SHAP library to obtain detailied information about feature importance for the models. |
| [subset_of_features.ipynb](./subset_of_features.ipynb)  | Trains a LightGBM model over the features extracted from our own Essentia extractor. |