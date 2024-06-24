#!/bin/bash

PICKLE_PATH=./results_feat_rnn/model_experiment.pkl
SAVE_PATH=./results_feat_rnn

~/anaconda3/bin/python3 rnn_hyperparameter_tuning.py $PICKLE_PATH $SAVE_PATH