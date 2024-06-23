import argparse
import os

from pydub import AudioSegment


def convert_mp3_to_wav(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Traverse the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
            mp3_path = os.path.join(input_folder, filename)
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(output_folder, wav_filename)

            # Load the MP3 file
            audio = AudioSegment.from_mp3(mp3_path)

            # Export as WAV
            audio.export(wav_path, format='wav')
            print(f"Converted {filename} to {wav_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP3 files to WAV format."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing MP3 files."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder where WAV files will be saved."
    )

    # Parse arguments
    args = parser.parse_args()

    # Convert MP3 files to WAV files
    convert_mp3_to_wav(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()
