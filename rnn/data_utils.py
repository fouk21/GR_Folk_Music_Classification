import numpy as np
import os
import torch
from sound_utils import SoundUtils
from torch.utils.data import Dataset

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
        return np.asarray(data), np.asarray(labels)
    
    @staticmethod
    def preprocess_feature_data():
        pass

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