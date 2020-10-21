import sys
from src.data.creation.influence_graph import Rels
from src.data.creation.influence_graph import SourceTextFormatter
import json
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
import sys
import re
from docopt import docopt
from typing import List, Tuple, Dict
import pandas as pd

add_reverse = True

relations = [Rels.special_tokens['helps'],
                    Rels.special_tokens['hurts']]

if add_reverse:
    relations.extend([Rels.special_tokens['helped_by'],
                            Rels.special_tokens['hurt_by']])


def clean(pth: str):
    data = pd.read_json(pth, orient="records", lines=True)
    data["predicted_answer_orig"] = data["predicted_answer"]
    data["answer_orig"] = data["answer"]
    data["predicted_answer"] = data["predicted_answer"].apply(lambda x: clean_output(x))
    data["answer"] = data["answer"].apply(lambda x: clean_output(x))
    data.to_json(f"{pth}.cleaned", orient="records", lines=True)

def clean_output(text: str) -> str:

    for rel in relations:
        pos = text.find(rel)
        if pos != -1:
            return text[pos + len(rel) + 1:]

    return text  # nothing to clean

if __name__ == "__main__":
    clean(sys.argv[1])