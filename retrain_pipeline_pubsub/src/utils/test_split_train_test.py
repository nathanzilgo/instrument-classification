import pandas as pd

from retrain_pipeline_pubsub.src.utils.split_train_test import split_train_test


def test_split_train_test():
    data = {
        'filename': ['file1', 'file2', 'file3'],
        'frame': [100, 200, 300],
        'track_id': [1, 2, 3],
        'label': [0, 1, 0],
        'dataset': ['train', 'test', 'train'],
    }
    features_df = pd.DataFrame(data)

    X_train, y_train, X_test, y_test = split_train_test(features_df.copy())

    assert len(X_train) == 2
    assert len(y_train) == 2
    assert len(X_test) == 1
    assert len(y_test) == 1
