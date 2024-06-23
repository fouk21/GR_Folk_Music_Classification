#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def load_model() -> dict:
    model_path = f'{CURRENT_DIR}/svm.joblib'

    # Load the SVM model
    svm_model = joblib.load(model_path)

    return svm_model


def format_input(df: pd.DataFrame) -> dict:
    X = df.drop('label', axis=1)
    labels = df['label']

    # Create a label encoder object
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return (X, y, labels, label_encoder)


def plot_cm(cm, labels):
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, square=True,
                xticklabels=np.unique(labels), yticklabels=np.unique(labels))

    # Labels, title, and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(np.unique(labels))
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_ticklabels(np.unique(labels))
    ax.yaxis.set_tick_params(rotation=0)
    plt.show()


def main():
    svm_model = load_model()

    # Load test dataset
    test_data_path = f'{CURRENT_DIR}/svm_test.csv'
    test_df = pd.read_csv(test_data_path)

    X_test, y_test_encoded, y_test_labels, label_encoder = format_input(
        test_df
    )

    # Make predictions on the test dataset
    y_pred_encoded = svm_model.predict(X_test)

    # Decode predictions back to original labels for consistency
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    # Evaluate the model performance
    report = classification_report(y_test_labels, y_pred_labels)

    # Display the results
    print('Classification Report:')
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    plot_cm(cm, label_encoder.classes_)


if __name__ == '__main__':
    main()
