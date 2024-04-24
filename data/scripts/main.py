import pandas as pd

from youtube_downloader import YouTubeDownloader


def main():
    df = pd.read_csv('raw.tsv', sep='\t')

    subset_df = df[['id', 'youtube-id', 'start-ts', 'end-ts']]
    song_list = subset_df.to_numpy()
    yt = YouTubeDownloader()
    yt.download(song_list)
    yt.cleanup()


if __name__ == '__main__':
    main()
