import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from typing import Dict


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


def plot_confusion_matrix(y_true, y_pred, labels=None, perc=True) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if perc:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(len(labels), len(labels)))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2%' if perc else 'd',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def print_classification_report(y_true, y_pred, labels=None) -> None:
    print(classification_report(y_true, y_pred, labels=labels))
