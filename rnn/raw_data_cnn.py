from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sound_utils import *
from data_utils import *
from model import *
from model_utils import ModelUtils
import pandas as pd

##############################################################
##############################################################

# SEED = 13 # reproducible results: Same results in every run
# IN_PATH = ''
# DATA_PATH = '' 
# OUT_PATH = ''
# EPOCH = 20 # number of epochs to run for model

# np.random.seed(SEED) 
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True  # cuda algorithms
# os.environ['PYTHONHASHSEED'] = str(SEED)


##############################################################
##############################################################


# Example usage:
wav_folder = '../data/cropped'
frame_length = 0  # Example: 1 second at 44.1 kHz sampling rate
max_lentgh = 9500000
data_df = pd.read_csv('dummy_dataset.csv')
num_classes = data_df['region'].nunique()  # Example: 3 classes for classification

# Load data for training
data, labels = DataUtils.load_raw_data_for_training(wav_folder, frame_length, data_df, max_lentgh, True)
# Calculate label proportions
unique_labels, label_counts = np.unique(labels, return_counts=True)
label_proportions = label_counts / len(labels)

# Split the data with balanced label distribution
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, train_size=0.7, stratify=labels)
remaining_size = len(temp_data)
val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, stratify=temp_labels)

train_dataset = CNNAudioDataset(train_data, train_labels)
val_dataset = CNNAudioDataset(val_data, val_labels)
test_dataset = CNNAudioDataset(test_data, test_labels)

batch_size = 2

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_shape = data.shape[1]

model_experiment = ModelUtils(train_loader,val_loader,test_loader,input_shape,num_classes)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

epochs = 1

input_shape = ()
path = ''
model_experiment.model_summary(test_loader, 'raw_data_cnn_model_summary', input_shape, path)

model_experiment.train_cnn(epochs)

model_experiment.test_cnn() #TODO: add path

model_experiment.save_model() #TODO: add path