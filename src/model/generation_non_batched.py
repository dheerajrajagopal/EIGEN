#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import torch
import sys
sys.path.insert(0, '..')


from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from typing import List, Dict
import itertools


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}


def get_predictions(args, model, tokenizer, input_sentence):
    encoded_prompt = tokenizer.encode(input_sentence, add_special_tokens=False, return_tensors='pt')
    encoded_prompt = encoded_prompt.cuda()
    output_dict = {'answer': ''}
    with torch.no_grad():
        out = model.generate(
            # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
            input_ids=encoded_prompt,
            max_length=args.max_len + encoded_prompt.size(-1),
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
        )

        for out_seq in out:
            text = tokenizer.decode(out_seq, clean_up_tokenization_spaces=True)
            text = text[: text.find(args.stop_token) if args.stop_token else None]
            answer = text[text.find('?') + 1:]
            output_dict['answer'] += answer
            output_dict['answer'] += '. '

        return output_dict


def read_jsonl(input_file: str) -> List[Dict]:
    output: List[Dict] = []
    with open(input_file, 'r') as open_file:
        for line in open_file:
            output.append(json.loads(line))
    return output


def aggregate_predictions(prediction_fp: str, out_fp: str, separator="||"):
    json_lines = read_jsonl(prediction_fp)

    with open(out_fp, 'w') as outfile:
        for grp_id, grp in itertools.groupby(json_lines, lambda line: separator.join(line["id"].split(separator)[0:2])):
            fmt_j = {"id": grp_id, "answers": []}
            for line in grp:
                # This is in the format required by the evaluator.
                fmt_j["answers"].append(line["answer"])
            print(json.dumps(fmt_j), file=outfile)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--max_len",
        default=20,
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--stop_token",
        type=str,
        default='<|endoftext|>',
        help="model path",
    )

    parser.add_argument(
        "--test_input_file",
        default=None,
        type=str,
        help="model path",
    )

    parser.add_argument(
        "--test_output_file",
        default=None,
        type=str,
        help="model path",
    )

    parser.add_argument(
        "--test_output_agg_file",
        default=None,
        type=str,
        help="model path",
    )

    args = parser.parse_args()

    config = GPT2Config.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.cuda()
    model.eval()


    test_input = []
    with open(args.test_input_file, 'r') as open_file:
        for line in open_file:
            test_input.append(json.loads(line))

    # In[18]:


    with open(args.test_output_file, 'w') as open_file:
        for item in test_input:
            output = get_predictions(args, model, tokenizer, item['question'])
            output['id'] = item['id']
            json.dump(output, open_file)
            open_file.write('\n')



    aggregate_predictions(args.test_output_file,
                          args.test_output_agg_file)

if __name__ == "__main__":
    main()
