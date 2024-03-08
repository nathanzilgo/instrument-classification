import pandas as pd


def split_train_test(features_extracted: pd.DataFrame):
    train_features = features_extracted[
        features_extracted['dataset'] != 'test'
    ]
    validation_features = features_extracted[
        features_extracted['dataset'] == 'test'
    ]

    X_train = train_features.drop(
        ['filename', 'frame', 'track_id', 'label', 'dataset'], axis=1
    ).to_numpy()
    y_train = train_features['label'].to_numpy()

    X_test = validation_features.drop(
        ['filename', 'frame', 'track_id', 'label', 'dataset'], axis=1
    ).to_numpy()
    y_test = validation_features['label'].to_numpy()

    return (X_train, y_train, X_test, y_test)
