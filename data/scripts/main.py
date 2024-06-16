import argparse
import os
import pandas as pd
import sys
import uuid

from youtube_downloader import YouTubeDownloader


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def download_dataset(file):
    df = None

    if file == 'lyra':
        # Lyra dataset
        url = 'https://raw.githubusercontent.com/pxaris/lyra-dataset/main/data/raw.tsv'
        df = pd.read_csv(url, sep='\t')
    elif file == 'enhanced':
        # Enhanced dataset
        path = f'{CURRENT_DIR}/../../data_exploration/new_data_v2.csv'
        df = pd.read_csv(path, sep=';')
        df['id'] = df['id'].apply(lambda _: f'{uuid.uuid4()}')
    else:
        df = pd.read_csv(file, sep=',')

    subset_df = df[['id', 'youtube-id', 'start-ts', 'end-ts']]
    subset_df.to_csv(f'{CURRENT_DIR}/{file}.csv', index=False)
    song_list = subset_df.to_numpy()

    yt = YouTubeDownloader()
    yt.download(song_list)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="File parameter")
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        required=True,
        help='Path or URL to the file',
    )

    # Parse the arguments
    args = parser.parse_args()
    file = args.file

    # Print the file path
    print(f"Received file: {file}")

    # Check if the file exists
    try:
        download_dataset(file)
    except FileNotFoundError:
        print(f"Error: The file '{file}' does not exist.")
        sys.exit(1)


if __name__ == '__main__':
    main()
