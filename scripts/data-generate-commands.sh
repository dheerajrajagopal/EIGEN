python src/data/creation/influence_graph.py --add_start_end True --add_reversed True --add_entire_path False --add_paragraph True --path_to_qa data/raw_data/qa/ --outpath data/nat_1 --path_to_influence_graphs data/raw_data/qa/wiqa_influence_graphs.jsonl --generation-type question

python src/data/creation/influence_graph.py --add_start_end True --add_reversed True --add_entire_path False --add_paragraph True --path_to_qa data/raw_data/qa/ --outpath data/nat_2 --path_to_influence_graphs data/raw_data/qa/wiqa_influence_graphs.jsonl --generation-type natural

python src/data/creation/influence_graph.py --add_start_end True --add_reversed False --add_entire_path False --add_paragraph True --path_to_qa data/raw_data/qa/ --outpath data/nat_3 --path_to_influence_graphs data/raw_data/qa/wiqa_influence_graphs.jsonl --generation-type question

python src/data/creation/influence_graph.py --add_start_end True --add_reversed False --add_entire_path False --add_paragraph True --path_to_qa data/raw_data/qa/ --outpath data/nat_4 --path_to_influence_graphs data/raw_data/qa/wiqa_influence_graphs.jsonl --generation-type natural

python src/data/creation/influence_graph.py --add_start_end True --add_reversed True --add_entire_path False --add_paragraph False --path_to_qa data/raw_data/qa/ --outpath data/nat_5 --path_to_influence_graphs data/raw_data/qa/wiqa_influence_graphs.jsonl --generation-type question

python src/data/creation/influence_graph.py --add_start_end True --add_reversed True --add_entire_path False --add_paragraph False --path_to_qa data/raw_data/qa/ --outpath data/nat_6 --path_to_influence_graphs data/raw_data/qa/wiqa_influence_graphs.jsonl --generation-type natural

python src/data/creation/influence_graph.py --add_start_end True --add_reversed False --add_entire_path False --add_paragraph False --path_to_qa data/raw_data/qa/ --outpath data/nat_7 --path_to_influence_graphs data/raw_data/qa/wiqa_influence_graphs.jsonl --generation-type question

python src/data/creation/influence_graph.py --add_start_end True --add_reversed False --add_entire_path False --add_paragraph False --path_to_qa data/raw_data/qa/ --outpath data/nat_8 --path_to_influence_graphs data/raw_data/qa/wiqa_influence_graphs.jsonl --generation-type natural
