#!/bin/bash

echo "Runing cpp test"

CASE_DIR=$(pwd)

export LD_LIBRARY_PATH=/home/su_estupinan@private.list.lu/Programs/libtorch//lib:$LD_LIBRARY_PATH

/home/su_estupinan@private.list.lu/Programs/XDEM/Cases/CPP_test/test_surrogate_VERTEX_overlapOnly

source /home/su_estupinan@private.list.lu/Programs/env_folder/bin/activate
python /home/su_estupinan@private.list.lu/Programs/XDEM/Cases/CPP_test/plot_predictions.py "$CASE_DIR"
