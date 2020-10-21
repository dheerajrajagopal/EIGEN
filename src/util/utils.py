import json
from pathlib import Path
from typing import Dict, List


def read_jsonl(input_file: str) -> List[Dict]:
    output: List[Dict] = []
    with open(input_file, 'r') as open_file:
        for line in open_file:
            output.append(json.loads(line))
    return output


def ensure_path(output_fp: str):
    # Create parent level subdirectories if not exists.
    p = Path(output_fp)
    if p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)