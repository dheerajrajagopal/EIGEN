#!/bin/bash
set -u
split="$1"
MODEL="$2"
OUTDIR="$3"

INPUT="data/qa/qa-${split}.jsonl"

mkdir -p "data/${OUTDIR}/"

OUTPATH="data/${OUTDIR}/qa-${split}.jsonl"


#MODEL="output/wiqa-grammar-rel-middle/checkpoint-12.48013-150000/"
#MODEL="output/wiqa-grammar-rel-middle/checkpoint-247.967-355000/"

CUDA_VISIBLE_DEVICES="$4" python -u src/eval/generate_context_prompt.py --model-path "${MODEL}"\
                                                                 --qa-file-path "${INPUT}"\
                                                                 --outpath "${OUTPATH}"\
                                                                 --graph-file-path data/qa/wiqa_influence_graphs.jsonl
