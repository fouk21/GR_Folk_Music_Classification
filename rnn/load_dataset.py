import numpy as np
import pandas as pd
from ast import literal_eval


# Function to convert string to numpy array
def string_to_array(string):
    # Remove leading and trailing brackets and quotes
    string = string.strip('"[]')
    # Split the string by spaces and convert to float
    array = np.array([float(x) for x in string.split()])
    return array


def main():
    literals = {
        'mfcc':  literal_eval,
        'chroma': literal_eval
    }
    df = pd.read_csv('rnn_features.csv', converters=literals)

    df.groupby('label').size()

    df['spectral_centroid'] = df['spectral_centroid'].apply(string_to_array)
    df['zcr'] = df['zcr'].apply(string_to_array)
    df['rms'] = df['rms'].apply(string_to_array)

    df['mfcc'] = df['mfcc'].apply(lambda x: np.array(x))
    df['chroma'] = df['chroma'].apply(lambda x: np.array(x))
    df['spectral_centroid'] = df['spectral_centroid'].apply(lambda x: np.array(x))
    df['zcr'] = df['zcr'].apply(lambda x: np.array(x))
    df['rms'] = df['rms'].apply(lambda x: np.array(x))


if __name__ == '__main__':
    main()
