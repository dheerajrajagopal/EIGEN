#!/bin/bash
OUTPATH="$1"
python src/data/synthesis/influence_graph.py --add_start_end True\
                                             --add_reversed True\
                                             --add_entire_path False\
                                             --add_paragraph True\
                                             --path_to_qa data/qa\
                                             --path_to_influence_graphs data/qa/wiqa_influence_graphs.jsonl\
                                             --outpath "${OUTPATH}"
