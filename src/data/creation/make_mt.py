"""Convert the data to a format suitable for mt models

Usage:
    python make_mt.py [options]

Options:
    --inpath=<str>              The path to the input files
    --outpath=<str>             The path to the output files
"""
import pandas as pd
from tqdm import tqdm
import json
import argparse
import docopt


def format_to_mt(inpath: str, outpath: str, file_prefix: str = "beforeafter"):
    def dump_mt(split: str):
        with open(f"{inpath}/{split}.jsonl", "r") as fin,\
            open(f"{outpath}/{file_prefix}_parallel.{split}.before", "w") as fout_before,\
            open(f"{outpath}/{file_prefix}_parallel.{split}.after", "w") as fout_after:
            for line in tqdm(fin):
                example = json.loads(line)
                fout_before.write(f"{example['question']}\n")
                fout_after.write(f"{example['answer']}\n")

    dump_mt("train")
    dump_mt("test")
    dump_mt("dev")
    
if __name__ == '__main__':
    import sys
    format_to_mt(inpath=sys.argv[1], outpath=sys.argv[2])