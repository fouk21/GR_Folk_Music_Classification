from itertools import cycle
from tqdm import tqdm
import torch
import numpy as np
from model import *
import matplotlib.pyplot as plt
import optuna
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from sklearn.preprocessing import label_binarize
import pickle
import plotly.io as pio

class ModelUtils():

    def __init__(self, train_loader, val_loader, test_loader, input_shape, num_classes, class_map) -> None:
        # Check for CUDA
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._train_dataloader = train_loader
        self._test_dataloader = test_loader
        self._validation_dataloader = val_loader
        self._epochs = 1
        self._optimizer = None
        self._criterion = None
        self._input_shape = input_shape
        self._numClasses = num_classes
        self.class_map = class_map
        self.decoding = {idx: label for label, idx in self.class_map.items()}

    @property
    def model(self):
        return self._model

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @criterion.setter
    def criterion(self, new_criterion):
        self._criterion = new_criterion

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    @model.setter
    def model(self, new_model):
        self._model = new_model

    def train_epoch(self, epoch_num, grad_clip=True):

        self._model.train()
        epoch_loss = 0
        epoch_acc = 0
        with tqdm(self._train_dataloader, unit="batch") as tepoch:  # Wrap the dataloader with tqdm
            tepoch.set_description(f"Epoch {epoch_num}/{self._epochs}")
            for batch_data, batch_labels, seq_lengths in tepoch:
                batch_data, batch_labels = batch_data.to(self._device), batch_labels.to(self._device)
                seq_lengths = seq_lengths.cpu()
                self._optimizer.zero_grad()
                outputs = self._model(batch_data,seq_lengths)
                #print(seq_lengths)
                pred_classes = torch.argmax(F.softmax(outputs, dim=1),dim=1)#torch.round(torch.sigmoid(predictions))
                # print(batch_labels.shape)
                correct_outputs = (pred_classes == batch_labels).float()
                acc = correct_outputs.sum() / len(correct_outputs)

                loss = self._criterion(outputs, batch_labels)

                loss.backward()

                if grad_clip:
                    for param in self._model.parameters():
                        if param.grad is None:
                            print("INFO: Grad Clip!")
                            continue
                        grad_val = torch.clamp(param.grad, -5, 5)

                self._optimizer.step()

                self._scheduler.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                tepoch.set_postfix(loss=loss.item())
        return epoch_loss/len(self._train_dataloader), epoch_acc/len(self._train_dataloader)

    def evaluate_epoch(self):

        total_loss = 0
        total_acc = 0

        self._model.eval()

        with torch.no_grad():

            for batch in self._validation_dataloader:

                data, labels, seq_lengths = batch[0], batch[1], batch[2]
                data, labels = data.to(self._device), labels.to(self._device)
                seq_lengths = seq_lengths.cpu()
                predictions = self._model(data,seq_lengths)
                loss = self._criterion(predictions,labels)

                pred_classes = torch.argmax(F.softmax(predictions, dim=1),dim=1)
                correct_preds = (pred_classes == labels).float()

                accuracy = correct_preds.sum()/len(correct_preds)

                total_loss += loss.item()
                total_acc += accuracy.item()

        return total_loss/len(self._validation_dataloader), total_acc/len(self._validation_dataloader)

    def print_model_metrics(self, y_test, y_pred_list, save_path):
        print("--------------------------------------------")
        print("Model confusion matrix: \n",confusion_matrix(y_test, y_pred_list))
        print("--------------------------------------------")
        y_true_labels = [self.decoding[num] for num in y_test]
        y_pred_labels = [self.decoding[num] for num in y_pred_list]
        print("Model classification report: \n",classification_report(y_true_labels, y_pred_labels,target_names=list(self.class_map.keys())))
        print("--------------------------------------------")
        print("Model accuracy: ",accuracy_score(y_test, y_pred_list))
        print("--------------------------------------------")

        with open(save_path+'metrics.txt', 'a') as file:
            print("--------------------------------------------",file=file)
            print("Model confusion matrix: \n",confusion_matrix(y_test, y_pred_list),file=file)
            print("--------------------------------------------",file=file)
            y_true_labels = [self.decoding[num] for num in y_test]
            y_pred_labels = [self.decoding[num] for num in y_pred_list]
            print("Model classification report: \n",classification_report(y_true_labels, y_pred_labels,target_names=list(self.class_map.keys())),file=file)
            print("--------------------------------------------",file=file)
            print("Model accuracy: ",accuracy_score(y_test, y_pred_list),file=file)
            print("--------------------------------------------",file=file)

    def plot_roc_curve(self, y_test, y_pred_prob_list, save_path):
        #  Convert to numpy array if it is not already
        y_pred_prob_list = np.array(y_pred_prob_list)
        y_test = np.array(y_test)

        y_test_bin = label_binarize(y_test, classes=np.arange(self._numClasses))
        # Initialize dictionaries to hold false positive rates, true positive rates, and ROC AUC values
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Compute ROC curve and ROC area for each class
        for i in range(self._numClasses):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob_list[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'purple', 'green', 'red', 'blue', 'cyan', 'magenta'])
        for i, color in zip(range(self._numClasses), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Multiclass')
        plt.legend(loc="lower right")
        plt.savefig(save_path+'roc_curve.png')
        plt.show()

#TODO: ADD CLASS MAPPING
    def test_model(self, path):

        y_test = []
        y_pred_list = []
        y_pred_prob_list = []
        self._model.eval()

        with torch.no_grad():

            for batch in self._test_dataloader:

                data, labels, seq_lengths = batch[0], batch[1], batch[2]
                data, labels= data.to(self._device), labels.to(self._device)
                seq_lengths = seq_lengths.cpu()
                predictions = self._model(data,seq_lengths)#.squeeze(1)
                predictions = predictions.cpu()
                pred_classes = torch.argmax(F.softmax(predictions, dim=1),dim=1)

                y_pred_list += pred_classes.tolist()
                y_pred_prob_list += predictions.tolist()

                y_test += labels.tolist()


        self.print_model_metrics(y_test, y_pred_list, path)

        self.plot_roc_curve(y_test, y_pred_prob_list, path)

    def train_evaluate(self, params, plot=False, save_path=None, epochs=1):

        print("INFO: Started Model Training")

        early_stopping = EarlyStopping(tolerance=2, min_delta=18)
        rnn_kwargs = {'num_layers':params['num_layers'],
                      'batch_first':True,
                      'dropout':params['dropout_probability']
                      }
        input_shape = self._input_shape
        drop = params['dropout_probability'] # how much to drop
        hidden_size = params['hidden_size']
        cell_type = params['cell_type']
        skip_connections = params['skip_connections']
        hidden_layers = params['hidden_layers']
        gradient_clipping = params['gradient_clipping']
        bidirectional = params['bidirectional']

        self._model = RNNModel(input_shape, self._numClasses, drop,
                               skip_connections, hidden_layers, hidden_size,
                               cell_type, bidirectional, **rnn_kwargs).to(self._device)

        self._criterion = torch.nn.CrossEntropyLoss()    #has softmax integrated
        lr = params['learning_rate'] #learning rate for optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(),lr=lr)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=10, gamma=0.1)
        #return
        train_losses, train_accs = [], []
        acc = []
        val_losses, val_accs = [], []
        for epoch in range(epochs):
            epoch_loss, epoch_acc = self.train_epoch(epoch+1,gradient_clipping)
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            val_loss,val_acc = self.evaluate_epoch()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            tqdm.write(f'''End of Epoch: {epoch+1}  |  Train Loss: {epoch_loss:.3f}  |  Val Loss: {val_loss:.3f}  |  Train Acc: {epoch_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%''')
            # early stopping
            early_stopping(epoch_loss, val_loss)
            if early_stopping.early_stop:
                print(f"\nINFO: Ending early. Converged in {epoch} epochs.")
                break

        if plot:
            print("Learning Curves of model:")
            plt.plot(train_accs, '-o')
            plt.plot(val_accs, '-o')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend(['Train', 'Validation'])
            plt.title('Train vs Validation Accuracy')
            if save_path:
                plt.savefig(save_path+'accuracy_curve.png')
            plt.show()

            plt.plot(train_losses, '-o')
            plt.plot(val_losses, '-o')
            plt.xlabel('epoch')
            plt.ylabel('losses')
            plt.legend(['Train', 'Validation'])
            plt.title('Train vs Validation Losses')
            if save_path:
                plt.savefig(save_path+'loss_curve.png')
            plt.show()

        #torch.save(self._model, 'model.pt')
        return val_acc

    def train_cnn(self, epochs):

        print("INFO: Model Training Started")
        self._model = CNNModel(self._numClasses)
        self._model.to(self._device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.0001)
        for epoch in range(1, epochs + 1):
            # Training phase
            self._model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(tqdm(self._train_dataloader, desc=f"Training Epoch {epoch}")):
                data = data.to(torch.float32)
                data, target = data.to(self._device), target.to(self._device)
                optimizer.zero_grad()
                output = self._model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self._train_dataloader)
            print(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')

            # Validation phase
            self._model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in tqdm(self._validation_dataloader, desc="Validating"):
                    data, target = data.to(self._device), target.to(self._device)
                    output = self._model(data)
                    val_loss += criterion(output, target).item()
                    # pred = output.argmax(dim=1, keepdim=True)
                    pred = torch.argmax(F.log_softmax(output, dim=-1),dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            avg_val_loss = val_loss / len(self._validation_dataloader)
            val_accuracy = correct / len(self._validation_dataloader)
            print(f'====> Validation set: Average loss: {avg_val_loss:.4f}, Accuracy: {correct}/{len(self._validation_dataloader)} ({100. * val_accuracy:.2f}%)')

    def test_cnn(self, path):

        y_test = []
        y_pred_list = []
        y_pred_prob_list = []
        self._model.eval()

        with torch.no_grad():

            for batch in self._test_dataloader:

                data, labels = batch[0], batch[1]
                data, labels = data.to(self._device), labels.to(self._device)
                predictions = self._model(data)
                predictions = predictions.cpu()
                pred_classes = torch.argmax(F.log_softmax(predictions, dim=-1),dim=1)

                y_pred_list += pred_classes.tolist()
                y_pred_prob_list += predictions.tolist()

                y_test += labels.tolist()

        self.print_model_metrics(y_test, y_pred_list, path)

        self.plot_roc_curve(y_test, y_pred_prob_list, path)

    def load_model(self, model_path):
        self._model = torch.load(model_path)
        print("INFO: Loaded model: ", model_path)

    def save_model(self, model_path):
        torch.save(self._model, model_path)
        print("INFO: Model Saved at: ", model_path)
        pass

    def model_summary(self, model_name, save_path):
        print("\n",self._model)
        with open(save_path+model_name+'.txt', 'a') as file:
            print(self._model, file=file)


class EarlyStopping:

    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


class ModelTuning:

    def __init__(self, model_experiment) -> None:

        self.model_experiment = model_experiment
        self.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), study_name='RNN Model Fine Tuning')

    def objective(self, trial):

        params = {
                'learning_rate': trial.suggest_categorical('learning_rate', [0.0001, 0.001]),
                'num_layers': trial.suggest_categorical('num_layers', [1, 2]),
                'skip_connections': trial.suggest_categorical('skip_connections', [True, False]),
                'dropout_probability': trial.suggest_categorical('dropout_probability', [0.0, 0.2, 0.5]),
                'hidden_layers': trial.suggest_categorical('hidden_layers', [1, 2]),
                'gradient_clipping': trial.suggest_categorical('gradient_clipping', [True, False]),
                'cell_type': trial.suggest_categorical('cell_type', ['lstm', 'gru', 'rnn']),
                'hidden_size': trial.suggest_categorical('hidden_size', [128, 64]),
                'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
                }

        return self.model_experiment.train_evaluate(params, False, None, 2)

    def finetune_model(self, n_trials, save_path):
        self.study.optimize(self.objective, n_trials=n_trials)

        self.best_trial = self.study.best_trial

        print("INFO: Best model hyperparameters:")
        for key, value in self.best_trial.params.items():
            print("{} : {}".format(key, value))

        with open(f'{save_path}/rnn_tuned.pkl', 'wb') as file:
            pickle.dump(self.best_trial.params, file)

    def plot_tuning(self, save_path):
        pio.write_image(
            plot_optimization_history(self.study),
            f'{save_path}/opti_hist.png'
        )
        pio.write_image(
            plot_parallel_coordinate(self.study),
            f'{save_path}/paral_coord.png'
        )
        pio.write_image(
            plot_param_importances(self.study),
            f'{save_path}/param_imp.png'
        )
        pio.write_image(
            plot_contour(self.study, params=['num_layers', 'hidden_size']),
            f'{save_path}/layers_hiden.png'
        )
        pio.write_image(
            plot_contour(self.study, params=['cell_type', 'num_layers']),
            f'{save_path}/cell_layers.png'
        )
        pio.write_image(
            plot_contour(self.study, params=['bidirectional', 'num_layers']),
            f'{save_path}/bide_layers.png'
        )
