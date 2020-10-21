"""Merge question files
"""
import pandas as pd
def merge_questions(*files)->pd.DataFrame:
    res = dict()
    print(files)

    for file in files:
        print(f"reading {file}")
        data = pd.read_json(file, orient="records", lines=True)
        for i, row in data.iterrows():
            res[row["metadata"]["ques_id"]] = row
    
    return pd.DataFrame(list(res.values()))

def clean_generations(*files):
    def clean_context(context: str) -> str:
        if "<NODE>" not in context:
            return context
        else:
            return context.split(".")[-2]

    for file in files:
        data = pd.read_json(file, orient="records", lines=True)
        res = []
        for i, row in data.iterrows():
            row["question"]["generated_effect_context"] = clean_context(row["generated_effect_context"])
            row["question"]["generated_cause_context"] = clean_context(row["generated_cause_context"])
            res.append(row)
        pd.DataFrame(res).to_json(file, orient="records", lines=True)            

if __name__ == '__main__':
    import sys

    # merged_questions = merge_questions(*sys.argv[1:])
    # merged_questions.to_json("qa-train-with-context.jsonl.all", orient="records", lines=True)
    clean_generations(*sys.argv[1:])