"""
Given a predictions file, adds a sentence level metric to 
each line
"""
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords

english_stopwords = set(stopwords.words('english'))

def run(pth: str):
    data = pd.read_json(pth, orient="records", lines=True)
    res = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        predicted_answer = row["predicted_answer"][:-1].split()
        true_answer = [row["answer"].split()]
        row["bleu"] = sentence_bleu(references=true_answer, hypothesis=predicted_answer)
        row["jaccard"] = sentence_jaccard(row["predicted_answer"][:-1], row["answer"])
        row["predicted_answer"] = row["predicted_answer"][:-1]
        res.append(row)
    # pd.DataFrame(res).to_json(f"{pth}.with_bleu", orient="records", lines=True)
    pd.DataFrame(res).to_csv(f"{pth}.with_bleu", index=None, sep="\t")


def sentence_jaccard(s1: str, s2: str) -> int:

    s1 = set(s1.split()) - english_stopwords
    s2 = set(s2.split()) - english_stopwords

    return round(len(s1.intersection(s2)) / len(s1.union(s2)), 4)


if __name__ == "__main__":
    import sys
    run(sys.argv[1])

