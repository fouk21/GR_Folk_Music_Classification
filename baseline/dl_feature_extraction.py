import librosa
import numpy as np
import os
import pandas as pd
import pickle


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_FOLDER = f'{CURRENT_DIR}/../data/musical_regions'


def downsample(factor, feat, target_frames=100):
    feat_len = len(feat)

    # Initialize the new feature array with the target shape
    new_feat = np.zeros((feat_len, target_frames))

    # No need for padding cause our songs will always be long enough
    for i in range(target_frames):
        start_idx = int(i * factor)
        end_idx = int((i + 1) * factor)
        new_feat[:, i] = np.mean(feat[:, start_idx:end_idx], axis=1)

    return new_feat


def extract_features(
    file_path,
    target_frames=100,
    frame_length_ms=20,
    hop_length_ms=10,
    sample_rate=22050,
):
    # Calculate window size and hop length
    window_size = int(sample_rate * frame_length_ms / 1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)

    # Load the audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Extract features
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=window_size,
        hop_length=hop_length,
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=window_size,
        hop_length=hop_length,
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=window_size,
        hop_length=hop_length,
    )

    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        hop_length=hop_length,
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=window_size,
        hop_length=hop_length,
    )

    # Determine the number of frames in the original MFCC
    mfcc_frames = mfcc.shape[1]
    chroma_frames = chroma.shape[1]
    spectral_centroid_frames = spectral_centroid.shape[1]
    zcr_frames = zcr.shape[1]
    rms_frames = rms.shape[1]

    # Calculate the averaging factor
    mfcc_factor = mfcc_frames // target_frames
    chroma_factor = chroma_frames // target_frames
    spectral_centroid_factor = spectral_centroid_frames // target_frames
    zcr_factor = zcr_frames // target_frames
    rms_factor = rms_frames // target_frames

    new_mfcc = downsample(mfcc_factor, mfcc)
    new_chroma = downsample(chroma_factor, chroma)
    new_spec_centroid = downsample(spectral_centroid_factor, spectral_centroid)
    new_zcr = downsample(zcr_factor, zcr)
    new_rms = downsample(rms_factor, rms)

    return (
        new_mfcc.tolist(),
        new_chroma.tolist(),
        new_spec_centroid,
        new_zcr,
        new_rms,
    )


def main():
    dataset = f'{CURRENT_DIR}/../data/scripts/gr_folk_music.csv'
    df = pd.read_csv(dataset)

    regions = np.sort(df['region'].dropna().unique())
    regions = regions[-5:]
    print(regions)

    features = []
    cols = [
        'id',
        'label',
        'mfcc',
        'chroma',
        'spectral_centroid',
        'zcr',
        'rms',
    ]

    for region in regions:
        i = 1
        path = f'{CLASS_FOLDER}/{region}'
        tmpDf = df[df['region'] == region]
        file_list = [x for x in tmpDf['id'].to_list()]
        total = len(file_list)

        for file in file_list:
            print(f'Analyzing features for {region}: {i:03} out of {total:03}')
            i += 1
            features.append(
                [file, region] + list(extract_features(f'{path}/{file}.mp3'))
            )

    # Save the array just in case
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)

    # Create the features dataframe
    df = pd.DataFrame(features, columns=cols)
    df.to_csv(f'{CURRENT_DIR}/rnn_features.csv', index=False)


if __name__ == '__main__':
    main()
