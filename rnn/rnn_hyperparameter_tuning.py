from model_utils import ModelTuning
import pickle
import argparse

def main():

    parser = argparse.ArgumentParser(description="Arguments for Model Hyperparameter Tuning")

    parser.add_argument('pickle_path', type=str, help='Folder with pickled model object')
    parser.add_argument('save_path', type=str, help='Folder to save pickled best_params dict')

    args = parser.parse_args()
    
    print("INFO: Started Model Hyperparameter Tuning")

    with open(args.pickle_path, 'rb') as file:
        model_class = pickle.load(file)

    tuning_experiment = ModelTuning(model_class)

    tuning_experiment.finetune_model(2,args.save_path)

    tuning_experiment.plot_tuning()



if __name__ == "__main__":
    main()