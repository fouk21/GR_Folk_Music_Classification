from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sound_utils import *
from data_utils import *
from model import *
from model_utils import ModelUtils
import pandas as pd
import argparse
import pickle

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


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


def main():

    parser = argparse.ArgumentParser(description="Arguments for RNN Model")

    parser.add_argument('wav_folder', type=str, help='Folder with raw audio files')
    parser.add_argument('max_length', type=int, help='Max audio lentgh')
    parser.add_argument('dataset_location', type=str, help='CSV Dataset location')
    parser.add_argument('frame_length', type=int, help='Lentgh of each audio segment')
    parser.add_argument('results_dir', type=str, help='Folder to store model metrics and results')
    parser.add_argument('batch_size', type=int, help='Batch size for data loader')
    parser.add_argument('epochs', type=int, help='Number of training epochs')

    args = parser.parse_args()

    wav_folder = args.wav_folder
    frame_length = args.frame_length    
    max_lentgh = args.max_length
    data_df = pd.read_csv(f'{CURRENT_DIR}/../data/scripts/{args.dataset_location}')
    batch_size = args.batch_size
    results_dir = args.results_dir
    epochs = args.epochs
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    num_classes = data_df['region'].nunique()

    # Load data for training
    data, labels, class_mapping = DataUtils.load_raw_data_for_training(wav_folder, frame_length, data_df, max_lentgh, True)
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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("INFO: Data loaded successfully")

    input_shape = data.shape[1]

    model_experiment = ModelUtils(train_loader,val_loader,test_loader,input_shape,num_classes,class_mapping)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_experiment.train_cnn(epochs)

    model_experiment.test_cnn(results_dir)

    model_experiment.model_summary('raw_data_rnn_model_summary', results_dir)

    model_experiment.save_model(results_dir+'model.pth')

    with open(results_dir+'model_experiment.pkl', 'wb') as file:
        pickle.dump(model_experiment, file)
        print("INFO: pickled model class at: ", results_dir+'model_experiment.pkl')    