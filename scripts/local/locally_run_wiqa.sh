#!/bin/bash

export EXPERIMENT_TYPE="$1"

#  eval file is a subset of dev file
export TRAIN_FILE=data/"${EXPERIMENT_TYPE}"/train.jsonl
export DEV_FILE=data/"${EXPERIMENT_TYPE}"/dev.jsonl
export EVAL_FILE=data/"${EXPERIMENT_TYPE}"/eval.jsonl
export TEST_FILE=data/"${EXPERIMENT_TYPE}"/test.jsonl
export OUTPUT_DIR="output/${EXPERIMENT_TYPE}"
mkdir -p "output/${EXPERIMENT_TYPE}"

CUDA_VISIBLE_DEVICES="$2" python run_eigen.py \
    --output_dir="${OUTPUT_DIR}"\
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --do_train \
    --train_data_file=$TRAIN_FILE \
  	--per_gpu_train_batch_size 1 \
  	--per_gpu_eval_batch_size 2 \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
    --do_test \
    --test_data_file=$TEST_FILE \
    --length 12 \
    --overwrite_output_dir \
    --block_size 300 \
    --evaluate_during_training \
    --num_train_epochs 5 \
    --overwrite_cache\
    --save_steps 5000\
    --logging_steps 5000
