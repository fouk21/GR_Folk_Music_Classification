import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    # Load the data
    df = pd.read_csv(f'{CURRENT_DIR}/ml_features.csv')

    # Separate features and labels
    features = df.drop('label', axis=1)
    labels = df['label']

    # Encode labels if they are not numeric
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Split the df into training+validation and test sets
    features_train_val, features_test, labels_train_val, labels_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42
    )

    # Save the test features and labels for later use
    test_data = pd.DataFrame(features_test)
    test_data['label'] = labels_test
    test_data.to_csv(f'{CURRENT_DIR}/ff_test.csv', index=False)

    # Split the training+validation set into training and validation sets
    f_array, f_val_array, labels, val_labels = train_test_split(
        features_train_val,
        labels_train_val,
        test_size=0.25,
        random_state=42
    )

    NUM_CLASSES = len(np.unique(labels))
    EPOCHS = 200
    BATCH_SIZE = 100

    print(NUM_CLASSES)

    # Define the model
    model = Sequential([
        Dense(64, input_dim=138, activation='relu'),
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax'),
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Train the model
    history = model.fit(
        f_array,
        labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(f_val_array, val_labels),
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(
        f_val_array,
        val_labels
    )

    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Save the model
    model.save(f'{CURRENT_DIR}/ff_model.keras')

    # Plotting the training history
    plt.figure(figsize=(12, 4))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
