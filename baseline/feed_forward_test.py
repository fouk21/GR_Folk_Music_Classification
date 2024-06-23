import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def plot_cm(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()


def load_and_test_model():
    # Load the test data
    test_df = pd.read_csv(f'{CURRENT_DIR}/ff_test.csv')

    # Separate features and labels
    test_features = test_df.drop('label', axis=1)
    test_labels = test_df['label']

    # Encode labels if they are not numeric
    label_encoder = LabelEncoder()
    test_labels_encoded = label_encoder.fit_transform(test_labels)
    class_names = label_encoder.classes_

    # Convert class_names to strings if they are not already
    class_names = [str(cls) for cls in class_names]

    # Load the model
    model = tf.keras.models.load_model(f'{CURRENT_DIR}/ff_model.keras')

    # Evaluate the model
    loss, accuracy = model.evaluate(
        test_features,
        test_labels_encoded
    )

    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # Predict the labels for the test set
    test_predictions = model.predict(test_features)
    test_predictions = np.argmax(test_predictions, axis=1)

    # Compute the confusion matrix
    cm = confusion_matrix(test_labels_encoded, test_predictions)

    # Plot the confusion matrix
    plot_cm(cm, class_names)

    # Print classification report
    print(classification_report(
        test_labels_encoded,
        test_predictions,
        target_names=class_names
    ))


if __name__ == '__main__':
    load_and_test_model()
