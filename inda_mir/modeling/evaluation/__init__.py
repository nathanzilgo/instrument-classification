import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
)

from typing import Dict, List

from inda_mir.modeling.models import BaseModel
from inda_mir.modeling.train_test_split import DatasetInterface


def cross_val_score(model: BaseModel, folds: List[DatasetInterface]):

    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    fimportance_scores = []

    for data in folds:
        X_train, y_train = data.get_numpy_train_data()
        X_test, y_test = data.get_numpy_test_data()

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, pred))
        recall_scores.append(recall_score(y_test, pred, average='weighted'))
        precision_scores.append(
            precision_score(y_test, pred, average='weighted')
        )
        f1_scores.append(f1_score(y_test, pred, average='weighted'))
        try:
            fimportance_scores.append(
                model.get_feature_importance(data.get_features_names())
            )
        except:
            pass

    scores = {
        'accuracy': accuracy_scores,
        'recall': recall_scores,
        'precision': precision_scores,
        'f1': f1_scores,
        'feature_importances': fimportance_scores,
    }

    return scores


def plot_feature_importance(
    feature_importances: Dict[str, float], k: int = 20
) -> None:
    # Extract names and values from the dictionary
    feature_importances = dict(
        sorted(
            feature_importances.items(), key=lambda item: item[1], reverse=True
        )
    )
    names = list(feature_importances.keys())[:k]
    values = list(feature_importances.values())[:k]

    # Create a bar plot
    plt.bar(names, values)

    # Adding labels and title
    plt.xlabel('Names')
    plt.ylabel('Values')
    plt.title('Bar Plot of Name vs. Value')

    # Rotate x-axis labels if they are too long
    plt.xticks(rotation=90, ha='right')

    # Show the plot
    plt.show()


def plot_confusion_matrix(y_true, X_test, model, perc=True) -> None:
    y_pred = model.predict(X_test)
    labels = model.classes_
    cm = confusion_matrix(y_true, y_pred, labels=labels)[: len(labels) - 1]

    if perc:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(len(labels), len(labels)))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%' if perc else 'd',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels[:-1],
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model.name}')
    plt.show()


def plot_confusion_matrix_tracklevel(
    model, predictions, y_test, features, threshold=0.7
):
    aux_dataset = features[['track_id']].copy()
    aux_dataset['truth'] = y_test
    aux_dataset['prediction'] = predictions

    class_by_tracks = (
        aux_dataset.groupby(['track_id'])
        .agg(
            {
                'truth': 'min',
                'prediction': lambda x: np.random.choice(x.mode(dropna=False)),
            }
        )
        .reset_index()
    )

    labels = model.classes_
    cm = confusion_matrix(
        class_by_tracks['truth'].to_numpy(),
        class_by_tracks['prediction'].to_numpy(),
        labels=labels,
    )[: len(labels) - 1]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(len(labels), len(labels)))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels[:-1],
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(
        f'Confusion Matrix - {model.name}, Confidence: {threshold*100}%. Track level.'
    )
    plt.show()


def print_classification_report(y_true, y_pred, labels=None) -> None:
    print(classification_report(y_true, y_pred, labels=labels))


def plot_model_comparison(model1, model2, X_test, y_test):
    # Evaluate models
    metrics_model1 = evaluate_model(model1, X_test, y_test)
    metrics_model2 = evaluate_model(model2, X_test, y_test)

    # Extract metric names
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Plot the bar graph
    bar_width = 0.35
    index = np.arange(len(metric_names))

    plt.bar(index, metrics_model1, bar_width, label=f'Model 1 {model1.name}')
    plt.bar(
        index + bar_width,
        metrics_model2,
        bar_width,
        label=f'Model 2 {model2.name}',
    )

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Comparison of Classification Models')
    plt.xticks(index + bar_width / 2, metric_names)
    plt.legend()
    plt.show()


def evaluate_model(model, X_test, y_test, threshold=0.7):
    # Make predictions on the test set
    y_pred = model.predict(X_test, threshold=threshold)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return accuracy, precision, recall, f1
