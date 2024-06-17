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
    def load_data_for_training(wav_folder, frame_length, fixed_length=None):
        file_names = os.listdir(wav_folder)
        file_paths = [os.path.join(wav_folder, file) for file in file_names]
        
        data = []
        labels = []
        
        for file_path in file_paths:
            audio_data = SoundUtils.load_wav(file_path, frame_length=frame_length, fixed_length=fixed_length)
            
            data.append(audio_data)  # Extend instead of append to flatten the list
            
            # Assuming labels based on file names or your dataset structure
            # Adjust as per your dataset
            # label = int(file_path.split('_')[1])  # Example: extracting label from filename
            labels.extend([0] * len(audio_data))  # Placeholder for labels, adjust accordingly
        #data = data / np.max(np.abs(data),axis=-1, keepdims=True)
        return np.asarray(data), np.asarray(labels)

    @staticmethod
    def preprocess_data(input_data, labels, frame_length):
        # Convert input_data to a numpy array
        input_data = np.array(input_data)
        
        #TODO: CHECK STEREO AUDIO
        # Example: Reshape if necessary (adjust according to your data structure)
        
        #input_data = input_data.reshape(-1, frame_length, 1)  # Assuming mono audio
        
        return input_data, labels

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
    
def collate_fn(batch):
    data, labels, lengths = zip(*batch)
    data = torch.stack(data)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    return data, labels, lengths