#!/bin/bash
set -u
python src/data/creation/influence_graph.py --add_start_end True\
       					    --add_reversed True\
					    --add_entire_path False\
					    --add_paragraph False\
					    --path_to_qa data/qa/\
					    --outpath data/nat_14\
					    --path_to_influence_graphs data/qa/wiqa_influence_graphs.jsonl\
					    --generation-type question_with_prompt
