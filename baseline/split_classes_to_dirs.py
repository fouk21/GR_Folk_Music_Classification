import os
import pandas as pd
import shutil

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_FOLDER = 'data/musical_regions'


def main():
    absolute_path_class_dir = f'{CURRENT_DIR}/../{CLASS_FOLDER}'
    processed_dir = f'{CURRENT_DIR}/../data/scripts/cropped'
    df = pd.read_csv(f'{CURRENT_DIR}/../data_exploration/preprocessed_dataset.csv')
    regions = df['region'].unique()

    for region in regions:
        path = f'{absolute_path_class_dir}/{region}'
        if not os.path.exists(path):
            os.makedirs(path)

    for region in regions:
        tmpDf = df[df['region'] == region]
        file_list = [f'{x}.mp3' for x in tmpDf['id'].to_list()]

        for file in file_list:
            shutil.move(
                f'{processed_dir}/{file}',
                f'{absolute_path_class_dir}/{region}/{file}'
            )


if __name__ == '__main__':
    main()
