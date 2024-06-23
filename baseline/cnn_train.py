import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import torch
import torch.nn as nn
import visdom


from alexnet import AlexNet
from PIL import Image, ImageDraw
from simple_cnn import SimpleCNN
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from torchvision import datasets, transforms
# from torchviz import make_dot

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = f'{CURRENT_DIR}/../dataset/spectrograms/train'
IMAGE_PATH = f'{CURRENT_DIR}/../dataset/mel-spectrograms/train'


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
    # regions = regions[-2:]
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


# Function to calculate mean and std
def calculate_mean_std(BATCH_SIZE):
    # Define a simple transform to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the dataset without normalization
    dataset = datasets.ImageFolder(root=IMAGE_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in dataloader:
        # batch size (the last batch can have smaller size)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std


def data_exploration():
    dataset = f'{CURRENT_DIR}/../data/scripts/gr_folk_music.csv'
    df = pd.read_csv(dataset)

    spec_sample_plotting(df)
    class_dist(df)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def cnn_train():
    # Initialize Visdom
    vis = visdom.Visdom()

    # Define relevant variables for the DL task
    BATCH_SIZE = 64
    NUM_CLASSES = 21
    LEARNING_RATE = 0.001
    EPOCHS = 15
    TRAIN_SPLIT = 0.8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate mean and std for the dataset
    mean, std = calculate_mean_std(BATCH_SIZE)

    all_transforms = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
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

    model = AlexNet(NUM_CLASSES).to(device)
    model.apply(initialize_weights)
    # model = SimpleCNN(NUM_CLASSES).to(device)
    summary(model, (3, 1000, 600))

    # # Create a dummy input tensor with the appropriate dimensions
    # # batch size of 1, 3 channels, 1000x600 image
    # dummy_input = torch.randn(1, 3, 1000, 600).to(device)

    # # Generate the graph
    # output = model(dummy_input)
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.format = 'png'
    # dot.render = 'model_architecture'

    # Set Loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # Set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1,
    )

    # Initialize Visdom windows
    vis_loss = vis.line(X=[0], Y=[0], opts=dict(
        title='Training Loss',
        xlabel='Epoch',
        ylabel='Loss'
    ))
    vis_val_loss = vis.line(X=[0], Y=[0], opts=dict(
        title='Validation Loss',
        xlabel='Epoch',
        ylabel='Loss'
    ))
    vis_acc = vis.line(X=[0], Y=[0], opts=dict(
        title='Training Accuracy',
        xlabel='Epoch',
        ylabel='Accuracy'
    ))
    vis_val_acc = vis.line(X=[0], Y=[0], opts=dict(
        title='Validation Accuracy',
        xlabel='Epoch',
        ylabel='Accuracy'
    ))

    # Initialize lists to track accuracy
    train_accuracies = []
    val_accuracies = []

    total_step = len(train_loader)

    for epoch in range(EPOCHS):
        # Start timing for the epoch
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

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

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100.0 * correct / total
        train_accuracies.append(train_accuracy)

        scheduler.step()

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(
            epoch + 1,
            EPOCHS,
            running_loss/total_step,
            # loss.item(),
            train_accuracy,
        ))

        # Update Visdom plots
        vis.line(
            X=[epoch+1],
            Y=[running_loss/total_step],
            win=vis_loss,
            update='append',
        )
        vis.line(
            X=[epoch+1],
            Y=[train_accuracy],
            win=vis_acc,
            update='append',
        )

        # Validate the model
        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)

        # Calculate validation accuracy
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        epoch_end = time.time()

        print('Validation Loss after epoch [{}/{}]: {:.4f}, Accuracy: {:.2f}% ({:.2f}s)'.format(
            epoch + 1,
            EPOCHS,
            val_loss,
            val_accuracy,
            epoch_end - epoch_start,
        ))

        # Update Visdom plots
        vis.line(
            X=[epoch+1],
            Y=[val_loss],
            win=vis_val_loss,
            update='append',
        )
        vis.line(
            X=[epoch+1],
            Y=[val_accuracy],
            win=vis_val_acc,
            update='append',
        )

    # Save the trained model
    # torch.save(model.state_dict(), 'cnn_alex_spec.pth')
    torch.save(model.state_dict(), 'cnn_simple_spec.pth')
    # torch.save(model.state_dict(), 'cnn_alex_mel_spec.pth')
    # torch.save(model.state_dict(), 'cnn_simple_mel_spec.pth')


def main():
    # data_exploration()
    cnn_train()


if __name__ == '__main__':
    main()
