import pickle
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

#plt.style.use('seaborn') # use seaborn style plotting

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

    # Model parameters
    parser.add_argument('learning_rate', type=float, help='optimizer learing rate')
    parser.add_argument('num_layers', type=int, help='Number of rnn stacked layers')
    parser.add_argument('skip_connections', type=bool, help='Skip connections between layers')
    parser.add_argument('dropout_probability', type=float, help='Dropout')
    parser.add_argument('hidden_layers', type=int, help='Number of dense layers')
    parser.add_argument('gradient_clipping', type=bool, help='Perform grad clip at training')
    parser.add_argument('cell_type', type=str, help='type of RNN architecture')
    parser.add_argument('bidirectional', type=bool, help='number of directions for RNN')
    parser.add_argument('hidden_size', type=int, help='Number of neurons')
    parser.add_argument('epochs', type=int, help='Number of training epochs')

    args = parser.parse_args()

    wav_folder = args.wav_folder  #'../data/cropped'
    frame_length = args.frame_length    #44100  # Example: 1 second at 44.1 kHz sampling rate
    max_lentgh = args.max_length #9500000
    data_df = pd.read_csv(f'{CURRENT_DIR}/../data/scripts/{args.dataset_location}')

    num_classes = data_df['region'].nunique()  # Example: 3 classes for classification

    params = {
                'learning_rate': args.learning_rate, #0.0001,
                'num_layers':  args.num_layers, #2,
                'skip_connections': args.skip_connections, #False,
                'dropout_probability': args.dropout_probability, #0.5,
                'hidden_layers': args.hidden_layers, #2,
                'gradient_clipping': args.gradient_clipping, #False,
                'cell_type': args.cell_type, #'rnn',
                'hidden_size': args.hidden_size, #128,
                'bidirectional': args.bidirectional, #False
                }
    
    epochs = args.epochs
    batch_size = args.batch_size

    # Load data for training
    data, labels, class_mapping = DataUtils.load_raw_data_for_training(wav_folder, frame_length, data_df, max_lentgh)

    # Calculate label proportions
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_proportions = label_counts / len(labels)

    # Split the data with balanced label distribution
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, train_size=0.7, stratify=labels)
    remaining_size = len(temp_data)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, stratify=temp_labels)

    train_dataset = AudioDataset(train_data, train_labels)
    val_dataset = AudioDataset(val_data, val_labels)
    test_dataset = AudioDataset(test_data, test_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("INFO: Data loaded successfully")

    input_shape = data.shape[2]

    # RNN model instantiation
    model_experiment = ModelUtils(train_loader,val_loader,test_loader,input_shape,num_classes,class_mapping)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_experiment.train_evaluate(params,True,results_dir,epochs)

    model_experiment.test_model(results_dir)

    model_experiment.model_summary('raw_data_rnn_model_summary', results_dir)

    model_experiment.save_model(results_dir+'model.pth')

    with open(results_dir+'model_experiment.pkl', 'wb') as file:
        pickle.dump(model_experiment, file)
        print("INFO: pickled model class at: ", results_dir+'model_experiment.pkl')

if __name__ == "__main__":
    main()