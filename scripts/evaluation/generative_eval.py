"""Generative evaluation
"""
import pandas as pd
import numpy as np
import tempfile
import sys
import subprocess
from collections import Counter
from typing import List

gain_words = {"helps", "more", "higher", "increase", "increases", "stronger", "faster", "greater", "longer", "larger", "helping"}
loss_words = {"hurts", "less", "lower", "decrease", "decreases", "weaker", "slower", "smaller", "hurting", "softer", "fewer"}
POLARITY_GAIN = 1
POLARITY_LOSS = 2
POLARITY_NEUTRAL = 3

def get_polarity(sent):
    if any([x in loss_words for x in sent.lower().split()]):
        return POLARITY_LOSS
    elif any([x in gain_words for x in sent.lower().split()]):
        return POLARITY_GAIN
    else:
        return POLARITY_NEUTRAL
        
def main(pth):
    data = pd.read_json(pth, orient="records", lines=True)
    print("\n ==== Overall Evaluation ==== \n")
    eval(data)

    print("\n ==== By Length and Relation ==== \n")
    for reln in sorted(list(data["reln"].unique())):
        for path_len in sorted(list(data["path_length"].unique())):
            tmp = data[(data["path_length"] == path_len) & (data["reln"] == reln)]
            print(f"\n ==== {reln} {path_len} ({len(tmp)} samples) ==== \n") 
            
            eval(data[(data["path_length"] == path_len) & (data["reln"] == reln)])

def eval(data: pd.DataFrame):
    hypothesis = [x.lower().strip().replace("\n", " ") for x in list(data["predicted_answer"])]
    for i, h in enumerate(hypothesis):
        if h[-1] == ".":
            hypothesis[i] = h[:-1]
    # hypothesis = [h[:-1] for h in hypothesis if h[-1] == "." else h]
    references = [x.lower().strip() for x in list(data["answer"])]
    
    hypothesis_polarity = np.array([get_polarity(sent) for sent in hypothesis])
    references_polarity = np.array([get_polarity(sent) for sent in references])

    accuracy = (hypothesis_polarity == references_polarity).astype(int).sum()
    
    print(f"Polarity = {round(accuracy * 100 / len(hypothesis_polarity),2)}")

    nlg_eval(hypothesis, references)

def nlg_eval(hypothesis: List[str], references: List[str]):
    with tempfile.NamedTemporaryFile() as hypo, tempfile.NamedTemporaryFile() as ref:
        dump_to_file(hypo.name, hypothesis)
        dump_to_file(ref.name, references)
        subprocess.check_call(f"nlg-eval --hypothesis={hypo.name} --references={ref.name} --no-skipthoughts --no-glove",
                      shell=True)
    
def dump_to_file(pth, list):
    with open(pth, "w") as f:
        for ele in list:
            f.write(f"{ele}\n")
    

if __name__ == '__main__':
    main(sys.argv[1])

    
        
