#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time

from joblib import dump
from sklearn.metrics import (
    accuracy_score, auc,
    confusion_matrix, f1_score,
    matthews_corrcoef, precision_score,
    recall_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.svm import SVC

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_dataset() -> dict:
    dataset = f'{CURRENT_DIR}/train.csv'
    df = pd.read_csv(dataset)

    X = df.drop('label', axis=1)
    labels = df['label']

    # Create a label encoder object
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return (X, y, labels)


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


def plot_roc(y, y_test, y_score, labels):
    n_classes = len(np.unique(y))
    y_test = label_binarize(y_test, classes=np.unique(y))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Colors for the ROC curves
    # Choose a colormap
    colormap = plt.get_cmap('tab20')

    # Generate colors from the colormap
    colors = [colormap(i / n_classes) for i in range(n_classes)]

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label='ROC curve of class {0} (area = {1:0.2f})'.format(
                labels[i], roc_auc[i]
            )
        )
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC for {0} Classes'.format(n_classes))
    plt.legend(loc="lower right")
    plt.show()


def main():
    # Define custom logger
    svm_logger = logging.getLogger('svm_logger')
    svm_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s %(levelname)8s %(process)7d > %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
    )
    svm_logger.addHandler(handler)

    X, y, labels = get_dataset()

    print(X.shape)

    # breakdown to train/validation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    start = time.time()

    C = 3
    gamma = 0.01
    svm = SVC(
        C=C,
        kernel='rbf',
        gamma=gamma,
        probability=True,
        random_state=42,
        cache_size=800
    )

    fitted = svm.fit(X_train, y_train)

    end = time.time()
    svm_logger.info(f'SVM model trained in: {(end - start):0.2f}s')

    y_score = fitted.decision_function(X_test)
    y_pred = svm.predict(X_test)

    # Save SVM trained model
    dump(svm, f'{CURRENT_DIR}/svm.joblib')

    # METRICS
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    matthews = matthews_corrcoef(y_test, y_pred)

    svm_logger.info(f'C: {C}')
    svm_logger.info(f'Gamma: {gamma}')
    svm_logger.info(f'Accuracy: {accuracy:0.3f}')
    svm_logger.info(f'Precision: {precision:0.3f}')
    svm_logger.info(f'Recall: {recall:0.3f}')
    svm_logger.info(f'F1: {f1:0.3f}')
    svm_logger.info(f'Matthews Correlation Coefficient: {matthews:0.3f}')

    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, labels)

    plot_roc(y, y_test, y_score, np.unique(labels))


if __name__ == '__main__':
    main()
