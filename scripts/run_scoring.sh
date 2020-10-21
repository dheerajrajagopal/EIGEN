#!/bin/bash
RUN_TIME="para_reversed_multihop/"
MODEL_PTH="output/${RUN_TIME}/checkpoint-300000/"
#INPATH="data/${RUN_TIME}/"
INPATH="data/wiqa-para-first-reln-first/"
OUTPATH="output/${RUN_TIME}/"
CUDA_VISIBLE_DEVICES=2 python -u src/eval/score.py generate "$MODEL_PTH" "$INPATH" "$OUTPATH"
