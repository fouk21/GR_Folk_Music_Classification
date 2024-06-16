import os
import torch
import torch.nn as nn

from simple_cnn import SimpleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchviz import make_dot
from torchsummary import summary

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Define relevant variables for the DL task
batch_size = 64
num_classes = 4
learning_rate = 0.001
num_epochs = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

data_dir = f'{CURRENT_DIR}/../data/spectrograms'
dataset = datasets.ImageFolder(
    data_dir,
    transform=all_transforms
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

model = SimpleCNN(num_classes).to(device)
summary(model, (3, 1000, 600))

# Create a dummy input tensor with the appropriate dimensions
# batch size of 1, 3 channels, 1000x600 image
dummy_input = torch.randn(1, 3, 1000, 600).to(device)

# Generate the graph
output = model(dummy_input)
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render = 'model_architecture'

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=learning_rate,
#     weight_decay=0.005,
#     momentum=0.9,
# )

total_step = len(dataloader)

for epoch in range(num_epochs):
    model.train()

    # Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(dataloader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(
        epoch+1,
        num_epochs,
        loss.item(),
    ))

# Save the trained model
torch.save(model.state_dict(), 'cnn_model.pth')
