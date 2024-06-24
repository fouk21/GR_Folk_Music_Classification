#!/bin/bash

DATASET_LOCATION=rnn_features.csv
RESULTS_DIR=./results_feat_trans/
EPOCHS=1
BATCH_SIZE=10

~/anaconda3/bin/python3 features_transformer.py $DATASET_LOCATION $RESULTS_DIR $BATCH_SIZE $EPOCHS
