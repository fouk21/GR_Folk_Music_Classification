import numpy as np
import os
import torch
from sound_utils import SoundUtils
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval

# Function to convert string to numpy array
def string_to_array(string):
    # Remove leading and trailing brackets and quotes
    string = string.strip('"[]')
    # Split the string by spaces and convert to float
    array = np.array([float(x) for x in string.split()])
    return array



class DataUtils():

    def __init__(self) -> None:
        pass

    @staticmethod
    # Function to load WAV files and prepare data for training
    def load_raw_data_for_training(wav_folder, frame_length, data_df, fixed_length=None, noSegmentation=False):
        
        data = []
        labels = []

        unique_regions = data_df['region'].unique()
        class_mapping = {region: idx for idx, region in enumerate(unique_regions)}
        data_df['region'] = data_df['region'].map(class_mapping)

        ids = data_df['id'].to_list()
        labels = data_df['region'].to_list()
        if len(ids) != len(labels):
            print("Data and labels missmatch detected!")

        file_paths = [os.path.join(wav_folder,id+'.mp3') for id in ids]
            
        for file_path in file_paths:
            if noSegmentation:
                audio_data = SoundUtils.load_wav(file_path, frame_length=0, fixed_length=fixed_length)
            else:
                audio_data = SoundUtils.load_wav(file_path, frame_length=frame_length, fixed_length=fixed_length)
            data.append(audio_data)  # Extend instead of append to flatten the list
        #data = data / np.max(np.abs(data),axis=-1, keepdims=True)
        return np.asarray(data), np.asarray(labels), class_mapping
    
    @staticmethod
    def preprocess_feature_data(path):
        literals = {
            'mfcc':  literal_eval,
            'chroma': literal_eval
        }
        df = pd.read_csv(path, converters=literals)

        df.groupby('label').size()

        df['spectral_centroid'] = df['spectral_centroid'].apply(string_to_array)
        df['zcr'] = df['zcr'].apply(string_to_array)
        df['rms'] = df['rms'].apply(string_to_array)

        df['mfcc'] = df['mfcc'].apply(lambda x: np.array(x))
        df['chroma'] = df['chroma'].apply(lambda x: np.array(x))
        df['spectral_centroid'] = df['spectral_centroid'].apply(lambda x: np.array(x))
        df['zcr'] = df['zcr'].apply(lambda x: np.array(x))
        df['rms'] = df['rms'].apply(lambda x: np.array(x))

        df = df.drop(columns=['id'])
        
        dataset = df.values
        labels = dataset[:, 0]
        dataset = dataset[:, 1:]
        # Create an empty list to store the concatenated arrays
        concatenated_data = []

        # Iterate over each set of arrays in the dataset
        for array_set in dataset:
            reshaped_arrays = []
            for array in array_set:
                # Reshape the array if it is of shape (n,)
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                reshaped_arrays.append(array)
            
            # Concatenate all arrays in the set along the first dimension (axis 0)
            concatenated_array = np.concatenate(reshaped_arrays, axis=0)
            # Append the transposed concatenated array to the list
            concatenated_data.append(concatenated_array.T)

        # Convert the list of concatenated arrays to a numpy array
        final_data = np.array(concatenated_data)

        unique_labels = df['label'].unique()
        encoding = {label: idx for idx, label in enumerate(unique_labels)}
        
        labels_encoded = [0] * len(labels)
        for i,label in enumerate(labels):
            labels_encoded[i] = encoding[label]

        return final_data, np.array(labels_encoded), len(unique_labels), encoding
    
        

# Custom Dataset class
class AudioDataset(Dataset):
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.lenghts = self.calculate_lengths(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        length  = self.lenghts[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long), length
    
    def calculate_lengths(self, data):
        lengths = []
        for sample in data:
            # Initialize the length to zero
            seq_length = 0
            for window in sample:
                # Check if all elements in the window are zero
                if np.any(window != 0):
                    seq_length += 1
            lengths.append(seq_length)
        return lengths
    
class CNNAudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming data[idx] is of shape [AUDIO_LENGTH]
        # Reshape it to [1, AUDIO_LENGTH] for Conv1D input
        sample = self.data[idx].reshape(1, -1)  # Reshape to [1, AUDIO_LENGTH]
        return sample, self.labels[idx]
    
def collate_fn(batch):
    data, labels, lengths = zip(*batch)
    data = torch.stack(data)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    return data, labels, lengths