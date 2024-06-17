import numpy as np
import os
import pandas as pd


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    cols = [
        'id',
        'youtube-id',
        'region',
    ]

    # Lyra
    preprocessed_lyra = 'preprocessed_dataset.csv'
    lyra_path = f'{CURRENT_DIR}/../../data_exploration/{preprocessed_lyra}'
    lyra_df = pd.read_csv(lyra_path, sep=',')
    lyra_df = lyra_df[cols]

    # Enhanced
    enhanced_path = f'{CURRENT_DIR}/enhanced.csv'
    enhanced_df = pd.read_csv(enhanced_path, sep=',')
    enhanced_df = enhanced_df[cols]

    # Merge datasets
    merged = pd.concat([lyra_df, enhanced_df])
    merged['region'] = merged['region'].apply(
        lambda x: 'Mainland-Greece' if x is np.nan else x,
    )
    merged.to_csv(
        f'{CURRENT_DIR}/gr_folk_music.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
