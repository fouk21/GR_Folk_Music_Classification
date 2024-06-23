#!/bin/bash

WAV_FOLDER=../data/cropped
FRAME_LENGTH=44100
MAX_LENGTH=9500000
DATASET_LOCATION=dummy_dataset.csv
LEARNING_RATE=0.0001
NUM_LAYERS=2
SKIP_CONNECTIONS=False
DROPOUT=0.5
HIDDEN_LAYERS=2
GRAD_CLIPPING=False
CELL_TYPE=lstm
HIDDEN_SIZE=128
BIDIRECTIONAL=True
EPOCHS=1
RESULTS_DIR=./results/
BATCH_SIZE=2



python3 raw_data_rnn.py $WAV_FOLDER $MAX_LENGTH $DATASET_LOCATION $FRAME_LENGTH $RESULTS_DIR $BATCH_SIZE $LEARNING_RATE $NUM_LAYERS $SKIP_CONNECTIONS $DROPOUT $HIDDEN_LAYERS $GRAD_CLIPPING $CELL_TYPE $BIDIRECTIONAL $HIDDEN_SIZE $EPOCHS