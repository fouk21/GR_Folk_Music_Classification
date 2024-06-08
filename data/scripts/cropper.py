import os

from moviepy.editor import AudioFileClip


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class MrCropper:
    CROPPED_PATH = f'{CURRENT_DIR}/cropped2'

    def __init__(self) -> None:
        pass

    def crop(self, opts):
        file_path, start, end = opts

        # file name handling
        basename = os.path.basename(file_path)
        filenm_wo_ext = os.path.splitext(basename)[0]
        output_file = f'{self.CROPPED_PATH}/{filenm_wo_ext}.mp3'

        # load audio file
        audio = AudioFileClip(file_path)

        # Crop the audio
        song = audio.subclip(start, end)

        # Write the cropped audio to a file
        song.write_audiofile(output_file, codec='mp3')
