#!/bin/bash
# Runs generation in parts

set -u
split="$1"
MODEL="$2"
OUTDIR="$3"

GPU1=5
GPU2=7

INPUT="data/qa/qa-${split}.jsonl"

mkdir -p "data/${OUTDIR}/"

OUTPATH="data/${OUTDIR}/qa-${split}.jsonl"


#MODEL="output/wiqa-grammar-rel-middle/checkpoint-12.48013-150000/"
#MODEL="output/wiqa-grammar-rel-middle/checkpoint-247.967-355000/"

CUDA_VISIBLE_DEVICES="${GPU1}" python -u src/eval/generate_context_prompt.py --model-path "${MODEL}"\
                                                                 --qa-file-path "${INPUT}.0"\
                                                                 --outpath "${OUTPATH}.0"\
                                                                 --graph-file-path data/qa/wiqa_influence_graphs.jsonl & 
BACK_PID_P0=$!

CUDA_VISIBLE_DEVICES="${GPU1}" python -u src/eval/generate_context_prompt.py --model-path "${MODEL}"\
                                                                 --qa-file-path "${INPUT}.1"\
                                                                 --outpath "${OUTPATH}.1"\
                                                                 --graph-file-path data/qa/wiqa_influence_graphs.jsonl & 
BACK_PID_P1=$!

CUDA_VISIBLE_DEVICES="${GPU2}" python -u src/eval/generate_context_prompt.py --model-path "${MODEL}"\
                                                                 --qa-file-path "${INPUT}.2"\
                                                                 --outpath "${OUTPATH}.2"\
                                                                 --graph-file-path data/qa/wiqa_influence_graphs.jsonl & 
BACK_PID_P2=$!

CUDA_VISIBLE_DEVICES="${GPU2}" python -u src/eval/generate_context_prompt.py --model-path "${MODEL}"\
                                                                 --qa-file-path "${INPUT}.3"\
                                                                 --outpath "${OUTPATH}.3"\
                                                                 --graph-file-path data/qa/wiqa_influence_graphs.jsonl & 
BACK_PID_P3=$!

wait "${BACK_PID_P0}"
wait "${BACK_PID_P1}"
wait "${BACK_PID_P2}"
wait "${BACK_PID_P3}"

cat "${OUTPATH}.0" "${OUTPATH}.1" "${OUTPATH}.2" "${OUTPATH}.3" > "${OUTPATH}"