import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from alexnet import AlexNet
from PIL import Image, ImageDraw
from simple_cnn import SimpleCNN
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from torchvision import datasets, transforms
from torchviz import make_dot

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = f'{CURRENT_DIR}/../data/spectrograms'


def plot_fig(ids, title, nrows=5, ncols=15):
    fig, ax = plt.subplots(nrows, ncols, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.5, hspace=1)

    for i, j in enumerate(ids[:nrows*ncols]):
        fname = os.path.join(IMAGE_PATH, title, f'{j}.png')
        img = Image.open(fname)
        idcol = ImageDraw.Draw(img)
        idcol.rectangle(((0, 0), (95, 95)), outline='white')
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(np.array(img))
        plt.axis('off')

    plt.suptitle(title, y=0.94)


def spec_sample_plotting(df):
    regions = np.sort(df['region'].dropna().unique())
    regions = regions[-2:]
    print(regions)

    for region in regions:
        ids = df[df['region'] == region]['id'].to_list()
        plot_fig(ids, region, nrows=2, ncols=3)


def class_dist(df):
    counts = df['region'].value_counts()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Class Distribution')
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')
    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()

    plt.show()


def data_exploration():
    dataset = f'{CURRENT_DIR}/../data/scripts/gr_folk_music.csv'
    df = pd.read_csv(dataset)

    spec_sample_plotting(df)
    class_dist(df)


def cnn_train():
    # Define relevant variables for the DL task
    BATCH_SIZE = 64
    NUM_CLASSES = 4
    LEARNING_RATE = 0.001
    EPOCHS = 12
    TRAIN_SPLIT = 0.8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(
        root=IMAGE_PATH,
        transform=all_transforms
    )

    # Split dataset into training and validation sets
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = AlexNet(NUM_CLASSES).to(device)
    # model = SimpleCNN(NUM_CLASSES).to(device)
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
        lr=LEARNING_RATE,
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=learning_rate,
    #     weight_decay=0.005,
    #     momentum=0.9,
    # )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1,
    )

    total_step = len(train_loader)
    # total_step = len(dataloader)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Load in the data in batches using the train_loader object
        # for i, (images, labels) in enumerate(dataloader):
        for i, (images, labels) in enumerate(train_loader):
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

            running_loss += loss.item()

        scheduler.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(
            epoch + 1,
            EPOCHS,
            running_loss/total_step,
            # loss.item(),
        ))

        # Validate the model
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print('Validation Loss after epoch [{}/{}]: {:.4f}'.format(
            epoch + 1,
            EPOCHS,
            val_loss/len(val_loader),
        ))

    # Save the trained model
    torch.save(model.state_dict(), 'cnn_model.pth')


def main():
    data_exploration()
    cnn_train()


if __name__ == '__main__':
    main()
