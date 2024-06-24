import numpy as np
import os
import torch

from simple_cnn import SimpleCNN
from alexnet import AlexNet
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Define relevant variables for the ML task
batch_size = 64
num_classes = 19

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

data_dir = f'{CURRENT_DIR}/../dataset/spectrograms/test'
# data_dir = f'{CURRENT_DIR}/../dataset/mel-spectrograms/test'
test_dataset = datasets.ImageFolder(
    data_dir,
    transform=all_transforms
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Load the trained model
# model = SimpleCNN(num_classes)
model = AlexNet(num_classes)
# model.load_state_dict(torch.load('cnn_simple_spec.pth'))
# model.load_state_dict(torch.load('cnn_simple_mel.pth'))
model.load_state_dict(torch.load('cnn_alex_spec.pth'))
# model.load_state_dict(torch.load('cnn_alex_mel.pth'))
model = model.to(device)

# Set the model to evaluation mode
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient calculation
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy:.4f}')

# Calculate precision, recall, and F1-score
precision = precision_score(
    all_labels,
    all_preds,
    average='weighted',
    zero_division=0,
)
recall = recall_score(
    all_labels,
    all_preds,
    average='weighted',
    zero_division=0,
)
f1 = f1_score(
    all_labels,
    all_preds,
    average='weighted',
    zero_division=0,
)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Generate a classification report
report = classification_report(
    all_labels,
    all_preds,
    target_names=test_dataset.classes,
    zero_division=0,
)
print(report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print(conf_matrix)
