from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_utils import *
from model import *
from model_utils import *
import pandas as pd
import argparse

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
    parser = argparse.ArgumentParser(description="Arguments for Transformer Model")

    parser.add_argument('dataset_location', type=str, help='CSV Dataset location')
    parser.add_argument('results_dir', type=str, help='Folder to store model metrics and results')
    parser.add_argument('batch_size', type=int, help='Batch size for data loader')
    parser.add_argument('epochs', type=int, help='Number of training epochs')

    args = parser.parse_args()

    data, labels, num_classes, class_map = DataUtils.preprocess_feature_data(args.dataset_location)
    decoding = {idx: label for label, idx in class_map.items()}

    epochs = args.epochs
    batch_size = args.batch_size

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Parameters for the transformer
    feature_dim = 28  # Dimension of features per frame
    nhead = 7  # Number of attention heads
    num_encoder_layers = 4  # Number of transformer encoder layers
    dim_feedforward = 256  # Dimension of feedforward network
    dropout = 0.1  # Dropout rate

    model = TransformerModel(feature_dim, num_classes, nhead, num_encoder_layers, dim_feedforward, dropout)

    learning_rate = 1e-4

    train_model(model, train_loader, val_loader, epochs, learning_rate)

    test_model(model, test_loader, num_classes, results_dir, decoding, class_map)


if __name__ == "__main__":
    main()
