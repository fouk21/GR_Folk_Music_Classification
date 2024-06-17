from tqdm import tqdm
import torch
from model import *
import matplotlib.pyplot as plt
import optuna
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

class ModelUtils():

    def __init__(self, train_loader, val_loader, test_loader, input_shape, num_classes) -> None:
        # Check for CUDA
        #TODO: CHANGE MPS
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._model = None
        self._train_dataloader = train_loader
        self._test_dataloader = test_loader
        self._validation_dataloader = val_loader
        self._epochs = 1
        self._optimizer = None
        self._criterion = None
        self._input_shape = input_shape
        self._numClasses = num_classes

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
            tepoch.set_description(f"Epoch {epoch_num+1}/{self._epochs}")
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
                
                #TODO: CHECK GRAD CLIPING
                if grad_clip:
                    for param in self._model.parameters():
                        if param.grad is None:
                            print("grad_clip")
                            continue
                        grad_val = torch.clamp(param.grad, -5, 5)
                
                self._optimizer.step()
                
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

                pred_classes = torch.argmax(F.softmax(predictions, dim=1),dim=1)#torch.round(torch.sigmoid(predictions))
                # print(pred_classes.shape)
                # print(batch_labels.shape)
                correct_preds = (pred_classes == labels).float()

                accuracy = correct_preds.sum()/len(correct_preds)

                total_loss += loss.item() 
                total_acc += accuracy.item()

        return total_loss/len(self._validation_dataloader), total_acc/len(self._validation_dataloader)
    
    def test_model(self):
    
        y_test = []
        y_pred_list = []
        y_pred_prob_list = []
        self._model.eval() # set the model in evaluation mode to not compute gradients and reduce overhead
        
        with torch.no_grad(): # turn of gradients calculation 
            
            for batch in self._test_dataloader:

                data, labels, seq_lengths = batch[0], batch[1], batch[2]
                data, labels= data.to(self._device), labels.to(self._device)
                seq_lengths = seq_lengths.cpu()
                predictions = self._model(data,seq_lengths)#.squeeze(1)
                predictions = predictions.cpu()
                pred_classes = torch.argmax(F.softmax(predictions, dim=1),dim=1)

                y_pred_list += pred_classes.tolist()
                #print(torch.max(predictions, dim=1).values)
                y_pred_prob_list += torch.max(predictions, dim=1).values.tolist()
                
                y_test += labels.tolist()
        
            
        print("--------------------------------------------")
        print("Model confusion matrix: \n",confusion_matrix(y_test, y_pred_list))
        print("--------------------------------------------")
        print("Model classification report: \n",classification_report(y_test, y_pred_list))
        print("--------------------------------------------")
        print("Model accuracy: ",accuracy_score(y_test, y_pred_list))
        print("--------------------------------------------")

        
        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred_prob_list)
        print("--------------------------------------------")
        print('roc_auc_score for model: ', roc_auc_score(y_test, y_pred_prob_list))
        print("--------------------------------------------")
        plt.title('Receiver Operating Characteristic - RNN')
        plt.plot(false_positive_rate, true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        

    def train_evaluate(self, params, plot=False, epochs=1):

        early_stopping = EarlyStopping(tolerance=2, min_delta=18)
        #TODO: ADD BIDIRECTIONAL TO PARAMS
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

        #TODO CHANGE LOSS FN
        self._criterion = torch.nn.CrossEntropyLoss()    #has sigmoid integrated
        #TODO: FIX LR TO BE PASSED DYNAMICALLY TO CLASS
        lr = params['learning_rate'] #learning rate for optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(),lr=lr) 
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
                print(f"\nEnding early. Converged in {epoch} epochs.")
                break
                
        if plot:
            print("Learning Curves of model:")        
            plt.plot(train_accs,'-o')
            plt.plot(val_accs,'-o')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend(['Train', 'Validation'])
            plt.title('Train vs Validation Accuracy')
            plt.show()

            plt.plot(train_losses,'-o')
            plt.plot(val_losses,'-o')
            plt.xlabel('epoch')
            plt.ylabel('losses')
            plt.legend(['Train', 'Validation'])
            plt.title('Train vs Validation Losses')
            plt.show()
        
        torch.save(self._model, 'model.pt')
        return self._model

class EarlyStopping:
    
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

#TODO: FIX TUNING
class ModelTuning:

    def __init__(self, model_experiment) -> None:
        
        self.model_experiment = model_experiment
        self.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), study_name='RNN Model Fine Tuning')

    def objective(self,trial):
    
        params = {
                'learning_rate': trial.suggest_categorical('learning_rate', [0.0001, 0.001]),
                'num_layers': trial.suggest_categorical('num_layers', [1, 2, 3]),
                'skip_connections': trial.suggest_categorical('skip_connections', [True, False]),
                'dropout_probability': trial.suggest_categorical('dropout_probability', [0.0, 0.2, 0.3]),
                'hidden_layers': trial.suggest_categorical('hidden_layers', [1, 2]),
                'gradient_clipping': trial.suggest_categorical('gradient_clipping', [True, False]),
                'cell_type': trial.suggest_categorical('cell_type', ['lstm', 'gru']),
                'hidden_size': trial.suggest_categorical('hidden_size', [128, 64])
                }
        
        return self.model_experiment.train_evaluate(params,False,1)
    
    def finetune_model(self):
        self.study.optimize(self.objective, n_trials=30)

        self.best_trial = self.study.best_trial

        print("Best model hyperparameters:")
        for key, value in self.best_trial.params.items():
            print("{} : {}".format(key, value))
    
    def plot_tuning(self):
        plot_optimization_history(self.study)
        plot_parallel_coordinate(self.study)
        optuna.visualization.plot_param_importances(self.study)
        optuna.visualization.plot_contour(self.study, params=['num_layers', 'hidden_size'])
        optuna.visualization.plot_contour(self.study, params=['cell_type', 'num_layers'])


#TODO: ADD LOGS TO FILES
#TODO: SAVE PLOTS TO FILES
#TODO: ADD MODEL SAVE

