from itertools import cycle
from matplotlib import pyplot as plt
from sklearn.metrics import auc, classification_report, roc_curve
import numpy as np
from torch import optim
import torch.nn as nn
import torch


def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels, _ in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        val_loss, val_acc = validate_model(model, val_loader)
        
        print(f"Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def validate_model(model, val_loader):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct_preds.double() / len(val_loader.dataset)
    
    return epoch_loss, epoch_acc

def plot_roc_curve(y_true, y_score, num_classes, save_path):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path+'model_roc_curve.png')
    plt.show()

def test_model(model, test_loader, num_classes, save_path, decoding, class_map):
    model.eval()
    correct_preds = 0
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = nn.Softmax(dim=1)(outputs)
            
            correct_preds += torch.sum(preds == labels.data)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_acc = correct_preds.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    y_true_labels = [decoding[num] for num in all_labels]
    y_pred_labels = [decoding[num] for num in all_preds]
    
    # Generate classification report
    print("Classification Report:")
    print("Model classification report: \n",classification_report(y_true_labels, y_pred_labels,target_names=list(class_map.keys())))
    with open(save_path+'model_metrics.txt', 'a') as file:
        print("Model classification report: \n",classification_report(y_true_labels, y_pred_labels,target_names=list(class_map.keys())),file=file)

    # Convert labels to one-hot encoding for ROC calculation
    all_labels_one_hot = np.eye(num_classes)[all_labels]
    plot_roc_curve(all_labels_one_hot, all_probs, num_classes, save_path)

