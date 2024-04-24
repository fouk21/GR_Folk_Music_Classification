import os
import shutil

from cropper import MrCropper
from pytube import YouTube


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class YouTubeDownloader:
    BASE_YT_URL = 'https://www.youtube.com/watch?v='
    RAW_PATH = f'{CURRENT_DIR}/raw'

    def __init__(self) -> None:
        self.cropper = MrCropper()

    def print_separator(self, e=None):
        if not e:
            print(e)
        print('--------------------------')

    def download(self, song_list: list):
        for nm, vID, s, e in song_list:
            new_file = f'{self.RAW_PATH}/{nm}.mp4'

            try:
                yt = YouTube(f'https://www.youtube.com/watch?v={vID}')
                audio = yt.streams.filter(only_audio=True).first()
                out_file = audio.download(output_path=self.RAW_PATH)
                os.rename(out_file, new_file)
            except Exception as e:
                self.print_separator(e)
                continue

            try:
                self.cropper.crop((new_file, s, e))
            except Exception:
                print(f'cropping failed for {nm}-{vID}')

            self.print_separator()

    def cleanup(self):
        for fileNm in os.listdir(self.RAW_PATH):
            file_path = os.path.join(self.RAW_PATH, fileNm)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
