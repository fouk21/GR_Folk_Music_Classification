import gc
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Spectrogram():
    def __init__(self, __audio_files) -> None:
        self.audio_files = __audio_files
        self.output_path = f'{CURRENT_DIR}/../data/spectrograms'
        os.makedirs(self.output_path, exist_ok=True)

    def _is_file(self, path):
        return os.path.isfile(path)

    def _normalize_signal(self, signal):
        signal = np.double(signal)
        signal = signal / (2.0 ** 15)
        return (signal - signal.mean()) /\
            ((np.abs(signal)).max() + 0.0000000001)

    def extract_spectrograms(self, region):
        i = 1
        total = len(self.audio_files)
        base_path = f'{self.output_path}/{region}'
        os.makedirs(base_path, exist_ok=True)

        for audio_file in self.audio_files:
            if not self._is_file(audio_file):
                continue

            print(f'Creating spectrogram for {region}: {i:02} out of {total}')
            i += 1

            base_name = os.path.basename(audio_file)
            base_name_wo_ext = os.path.splitext(base_name)[0]

            spec_name = f'{base_path}/{base_name_wo_ext}.png'

            if self._is_file(spec_name):
                print(f'Creating spectrogram for {region}: {i:02} out of {total}')
                continue

            # Load the audio file
            y, sr = librosa.load(audio_file, sr=None)

            # Normalize signal
            y = self._normalize_signal(y)

            # Compute the STFT
            D = librosa.stft(y)

            # Convert the amplitude to dB
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Plot and save the spectrogram
            # plt.figure(figsize=(10, 6))
            # librosa.display.specshow(
            #     S_db,
            #     sr=sr,
            #     x_axis='time',
            #     y_axis='log'
            # )
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('Spectrogram')
            # plt.savefig('spectrogram.png')
            # plt.close()

            # Plot and save the spectrogram without axis and details
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(
                S_db,
                sr=sr,
                x_axis=None,
                y_axis=None,
            )
            plt.gca().set_axis_off()
            plt.subplots_adjust(
                top=1,
                bottom=0,
                right=1,
                left=0,
                hspace=0,
                wspace=0,
            )
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(
                spec_name,
                bbox_inches='tight',
                pad_inches=0,
            )
            plt.close('all')

            del y
            del sr
            del D
            del S_db

            gc.collect()
