#!/bin/bash
split="$1"

INPUT="data/qa/qa-${split}.jsonl"

OUTDIR="$3"

mkdir -p "data/${OUTDIR}/"

OUTPATH="data/${OUTDIR}/qa-${split}.jsonl"

MODEL="$2"
#MODEL="output/wiqa-grammar-rel-middle/checkpoint-12.48013-150000/"
#MODEL="output/wiqa-grammar-rel-middle/checkpoint-247.967-355000/"

CUDA_VISIBLE_DEVICES="$4" python -u src/eval/generate_context.py --model-path "${MODEL}" --qa-file-path "${INPUT}" --outpath "${OUTPATH}"
