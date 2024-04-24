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
        if e:
            print(e)
        print('--------------------------')

    def download(self, song_list: list):
        for nm, vID, s, e in song_list:
            raw_file = f'{self.RAW_PATH}/{nm}.mp4'

            try:
                yt = YouTube(f'https://www.youtube.com/watch?v={vID}')
                audio = yt.streams.filter(only_audio=True).first()
                out_file = audio.download(output_path=self.RAW_PATH)
                os.rename(out_file, raw_file)
            except Exception as e:
                self.print_separator(e)
                continue

            try:
                self.cropper.crop((raw_file, s, e))
            except Exception:
                print(f'cropping failed for {nm}-{vID}')

            self._cleanup(raw_file)
            self.print_separator()

    def _cleanup(self, filename):
        try:
            if os.path.isfile(filename) or os.path.islink(filename):
                os.unlink(filename)
            elif os.path.isdir(filename):
                shutil.rmtree(filename)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (filename, e))
