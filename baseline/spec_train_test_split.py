import os
import shutil

from sklearn.model_selection import train_test_split


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def spec_train_test_split(data_dir, train_dir, test_dir, test_size=0.2):
    # Create train and test directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Iterate through each class directory in the data_dir
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            # List all files in the class directory
            files = os.listdir(class_path)
            train_files, test_files = train_test_split(
                files,
                test_size=test_size,
                random_state=42,
            )

            # Create class subdirectories in train and test directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            if not os.path.exists(train_class_dir):
                os.makedirs(train_class_dir)
            if not os.path.exists(test_class_dir):
                os.makedirs(test_class_dir)

            # Copy files to train directory
            for file in train_files:
                shutil.copy(
                    os.path.join(class_path, file),
                    os.path.join(train_class_dir, file),
                )

            # Copy files to test directory
            for file in test_files:
                shutil.copy(
                    os.path.join(class_path, file),
                    os.path.join(test_class_dir, file)
                )


def main():
    # Define directories
    data_dir = f'{CURRENT_DIR}/../data/mel-spectrograms'
    train_dir = f'{CURRENT_DIR}/../dataset/mel-spectrograms/train'
    test_dir = f'{CURRENT_DIR}/../dataset/mel-spectrograms/test'

    # Split the data
    spec_train_test_split(data_dir, train_dir, test_dir)


if __name__ == '__main__':
    main()
