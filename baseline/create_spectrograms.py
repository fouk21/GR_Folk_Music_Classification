import numpy as np
import os
import pandas as pd
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__))))

from helpers.spectrogram import Spectrogram

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_FOLDER = f'{CURRENT_DIR}/../data/musical_regions'


def file_list_abs_path(folder_path):
    # Get the absolute path of the folder
    folder_path = os.path.abspath(folder_path)

    # List all files in the folder with their absolute paths
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    return files


if __name__ == '__main__':
    dataset = f'{CURRENT_DIR}/../data/scripts/gr_folk_music.csv'
    df = pd.read_csv(dataset)

    regions = np.sort(df['region'].dropna().unique())
    # regions = regions[-2:]
    print(regions)

    features = []
    fMeanStd = []
    fns = []
    plots = []

    for region in regions:
        region_path = f'{CLASS_FOLDER}/{region}'
        print(region_path)

        file_list = file_list_abs_path(region_path)
        spec_extractor = Spectrogram(file_list)

        spec_extractor.extract_spectrograms(region)
