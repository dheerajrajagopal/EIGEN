# EIGEN models

Generating what-if graphs from text

## Training Data 

The training data can be downloaded [here](https://drive.google.com/file/d/1z0du5uQjZbb9Hv0fpuPl3X_zQGUKC5Jy/view?usp=sharing)

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


The QA dataset files are based on [WIQA](https://allenai.org/data/wiqa) dataset.

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
