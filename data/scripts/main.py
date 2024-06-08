import pandas as pd

from youtube_downloader import YouTubeDownloader


# def main():
#     url = 'https://raw.githubusercontent.com/pxaris/lyra-dataset/main/data/raw.tsv'
#     df = pd.read_csv(url, sep='\t')
#     subset_df = df[['id', 'youtube-id', 'start-ts', 'end-ts']]
#     song_list = subset_df.to_numpy()

#     yt = YouTubeDownloader()
#     yt.download(song_list)


def main():
    #url = 'https://raw.githubusercontent.com/pxaris/lyra-dataset/main/data/raw.tsv'
    df = pd.read_csv('../..//data_exploration/missing_songs.csv', sep=',')
    subset_df = df[['id', 'youtube-id', 'start-ts', 'end-ts']]
    song_list = subset_df.to_numpy()

    yt = YouTubeDownloader()
    yt.download(song_list)

if __name__ == '__main__':
    main()
