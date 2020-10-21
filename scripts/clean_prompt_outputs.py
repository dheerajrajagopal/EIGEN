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

{
            Rels.special_tokens['helps']: "HELPS",
            Rels.special_tokens['hurts']: "HURTS",
            Rels.special_tokens['hurt_by']: "X-RELATION-HURTS",
            Rels.special_tokens['helped_by']: "X-RELATION-HELPS"
        }

def clean(pth: str):
    
    data = pd.read_json(pth, orient="records", lines=True)
    res = []
    for i, row in data.iterrows():
        row["question"]["generated_cause_context_orig"] = row["question"]["generated_cause_context"]
        row["question"]["generated_effect_context_orig"] = row["question"]["generated_effect_context"]
        row["question"]["generated_cause_context"] = clean_output(row["question"]["generated_cause_context"])
        row["question"]["generated_effect_context"] = clean_output(row["question"]["generated_effect_context"])
        res.append(row)
    pd.DataFrame(res).to_json(f"{pth}.cleaned", orient="records", lines=True)

def clean_output(text: str) -> str:

    helps = text[:text.find("HURTS")]
    hurts = text[text.find("HURTS"):text.find("X-RELATION-HELPS")]
    x_helps = text[text.find("X-RELATION-HELPS"):text.find("X-RELATION-HURTS")]
    x_hurts = text[text.find("X-RELATION-HURTS"):]

    helps = clean_rel(helps, ["helps"])
    hurts = clean_rel(hurts, ["hurts"])
    x_helps = clean_rel(x_helps, ["is help",  "is helps", "is helps by", "is helped by"])
    x_hurts = clean_rel(x_hurts, ["is hurt", "is hurt by", "is hurts"])
    
    #  return f"HELPS {helps.strip()} HURTS {hurts.strip()} X-RELATION-HELPS {x_helps.strip()} X-RELATION-HURTS {x_hurts.strip()}"
    return f"HELPS {helps.strip()} HURTS {hurts.strip()}"


def clean_rel(text, rels):
    for rel in rels:
        pos = text.find(rel)
        if pos != -1:
            return text[pos + len(rel) + 1:]
    return text


if __name__ == "__main__":
    x = "HELPS the lung is deformed. helps a smaller amount of oxygen being delivered to the blood stream. HURTS the lung is deformed. hurts a greater amount of oxygen being delivered to the blood stream. X-RELATION-HELPS the lung is deformed. is helps a smaller amount of oxygen being delivered to the blood stream. X-RELATION-HURTS the lung is deformed. is hurts lung expands to make more oxygen."
    print(clean_output(x))
    clean(sys.argv[1])