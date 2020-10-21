# input = filepath (of the file dumped by predictor)
# id, ques, answer1
# id, ques, answer2

# output = filepath of aggregated outputs
# id, ques, [answer1, answer2]
import argparse
import itertools
import json

from src.util.utils import read_jsonl


def aggregate_predictions(prediction_fp: str, out_fp: str, separator="||"):
    json_lines = read_jsonl(prediction_fp)
    # example of an id is: "wikihow.com/...||1||1||2"
    #   might indicate wikihow article's sentence 1, entity 1, attribute 2
    # Aggregate by sentence (e.g., "wikihow.com/...||1) to get an array of state changes
    with open(out_fp, 'w') as outfile:
        for grp_id, grp in itertools.groupby(json_lines, lambda line: separator.join(line["id"].split(separator)[0:2])):
            fmt_j = {"id": grp_id, "answers": []}
            for line in grp:
                # This is in the format required by the evaluator.
                fmt_j["answers"].append(line["answer"])
            print(json.dumps(fmt_j), file=outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_path", "-i", required=True, type=str,
        help="Path to the unaggregated file e.g. data/trackworld/tw_bench/tw_bench_propara_npn_ea.jsonl."
    )
    parser.add_argument(
        "--output_path", "-o", required=True, type=str,
        help="Path to the aggregated file. e.g. data/trackworld/tw_bench/tw_bench_propara_npn_ea_aggregated.jsonl"
    )

    args = parser.parse_args()

    aggregate_predictions(prediction_fp=args.input_path,
                          out_fp=args.output_path)
    print(f"Output is in {args.output_path}")
