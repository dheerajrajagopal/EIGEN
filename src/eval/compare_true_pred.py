"""Compares the true and predicted output of gpt2
"""
import json
import sacrebleu
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
POLARITY_GAIN = "+"
POLARITY_LOSS = "-"
POLARITY_NEUTRAL = "o"

def main(inpath, split, suffix):
    print(f"{inpath} {split}")
    print(f"BLEU score = {compare_bleu(inpath, split, suffix)}")
    class_report, polarity_score_val = polarity_score(inpath, split, suffix)
    print(f"Polarity score = {polarity_score_val}")
    print(class_report)
    


def polarity_score(inpath: str, split: str, suffix: str) -> float:
    num_polarities_corr = 0
    total_lines = 0
    y_true, y_pred = [], []
    with open(f"{inpath}/{split}.jsonl{suffix}", 'r') as open_file:
        for line in open_file:
            input = json.loads(line)
            polarities = compare_polarities(
                true=input["answer"], pred=input["predicted_answer"])
            y_true.append(polarities["true_polarity"])
            y_pred.append(polarities["pred_polarity"])
            num_polarities_corr += polarities["is_same"]
            total_lines += 1
    return classification_report(y_true, y_pred, labels=[POLARITY_GAIN, POLARITY_LOSS, POLARITY_NEUTRAL]), round(num_polarities_corr * 100.0 / total_lines, 4)


def compare_polarities(true: str, pred: str) -> dict:
    """Compares the polarities of the true and predicted outcomes

    Arguments:
        true {[type]} -- [description]
        pred {[type]} -- [description]
    """
    gain_words = {"helps", "more", "higher", "increase", "stronger", "faster", "greater"}
    loss_words = {"hurts", "less", "lower", "decrease", "weaker", "slower"}


    def get_polarity(sent):
        if any([x in loss_words for x in sent.lower().split()]):
            return POLARITY_LOSS
        elif any([x in gain_words for x in sent.lower().split()]):
            return POLARITY_GAIN
        else:
            return POLARITY_NEUTRAL

    res = {"true_polarity": get_polarity(
        true), "pred_polarity": get_polarity(pred)}
    res["is_same"] = int(res["true_polarity"] == res["pred_polarity"])
    return res


def compare_bleu(inpath: str, split: str, suffix: str) -> float:
    """Calculates the BLEU of predicted and true answers

    Arguments:
        inpath {[str]} -- [description]
        split {[str]} -- [description]

    Returns:
        [type] -- [description]
    """

    sys, refs = [], []
    with open(f"{inpath}/{split}.jsonl{suffix}", 'r') as open_file:
        for line in open_file:
            input = json.loads(line)
            sys.append(input["predicted_answer"])
            refs.append(input["answer"])

    return round(sacrebleu.corpus_bleu(sys, ref_streams=[refs]).score, 4)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        suffix = sys.argv[3]
    else:
        suffix = ""
    main(sys.argv[1], sys.argv[2], suffix)