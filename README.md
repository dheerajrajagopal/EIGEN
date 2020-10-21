# WIQA gen models

Generating what-if graphs from text

## Training Data Creation

To create training data that can be used for training GPT-2, run the following command:

```sh
python src/data/creation/influence_graph.py\
        --add_start_end True\
        --add_reversed True\
        --add_entire_path False\
        --add_paragraph True\
        --path_to_qa data/qa\
        --para-at-end False\
        --reln-in-middle False\
        --simple-generation True\
        --outpath tmp/\
        --path_to_influence_graphs data/qa/wiqa_influence_graphs.jsonl
```

Options:

```stl
    --add_start_end=<BOOL>                  Whether to add examples that use start and end node
    --add_reversed=<BOOL>                   Should the nodes be reversed and added as training examples?
    --add_entire_path=<BOOL>                Add the entire path
    --add_paragraph=<BOOL>                  Should the paragraph be added to the src sentence?
    --para-at-end=<BOOL>                    Should the paragraph be added at the end?
    --reln-in-middle=<BOOL>                 If true, the relation is added to the middle (PARA-RELN-NODE)
    --simple-generation=<BOOL>              If true, the generations are in simpler format (PARA - RELN - NODE) without any keywords
    --path_to_qa=<str>                      Path to the directory that has qa files. This is used to create the train/test/dev splits
    --outpath=<str>                         Path where the output should be written
    --path_to_influence_graphs=<str>        Path to the influence graphs jsonl file
```

## Training

Please use scripts/local/locally_run_wiqa.sh.

## Generation (Inference)

```sh
python src/eval/generate.py --model-path CHECKPOINT-PATH --input-path INPATH --output-path OUTPATH
```

- CHECKPOINT-PATH: Model checkpoint path
- INPATH: A jsonl file. Each line should have a field "question" which is used to generate the output (i.e. the output is conditioned on the question field.)
- OUTPATH: Path to the output. The output file is the same as the input file with an added field: predicted_answer.

### Sample for generating dev outputs

```sh
python src/eval/generate.py --model-path output/wiqa-grammar-low-lr/checkpoint-18.50236-170000/ --input-path data/wiqa-grammar-low-lr/dev.jsonl --output-path tmp.jsonl
```

### Sample output

```json
{
    "para_id": "214",
    "graph_id": "25",
    "split": "dev",
    "reln": "RELATION-HELPS",
    "question": "RELATION-HELPS 2-HOP <NODE> during drought <PARA> Animals pull a fruit off a plant or pick a fruit up from the ground. Animals eat the fruit. Animals drop some seeds onto the ground. Animals eat some seeds. The seeds are in the animal's waste. The waste goes onto the ground. There are seeds on the ground in different areas away from the plant.",
    "answer": "less vegetation",
    "context": "Animals pull a fruit off a plant or pick a fruit up from the ground. Animals eat the fruit. Animals drop some seeds onto the ground. Animals eat some seeds. The seeds are in the animal's waste. The waste goes onto the ground. There are seeds on the ground in different areas away from the plant.",
    "id": 24,
    "predicted_answer": "LESS seeds being picked."
}
```


## Context Generation (For Training QA)

- In order to train downstream QA module, we need to add generated context to the QA files.

```sh
python -u src/eval/generate_context.py\
                --model-path CHECKPOINT\
                --qa-file-path QA-FILE\
                --outpath OUTPATH
```

Where QA-FILE is the original QA file.

- This file extracts the two nodes (n1, n2) from a given question. For each node, a context is generated using the model saved at CHECKPOINT.  
- The context for each node N has 4 nodes surrounding N with the following relations: HELPS, HURTS, HELPED-BY, and HURT-BY.
