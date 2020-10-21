"""Scores a given test/eval file

Usage:
    generate.py [options]

Options:
    --model-path=<str>          Path to the checkpoint
    --input-path=<str>          Path to the input file
    --output-path=<str>         Path to the output file
    --batch-size=<int>          The batch size [default: 32]

"""

from typing import List
from docopt import docopt
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import json
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
import sys
sys.path.insert(0, '..')
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}


class Gpt2Generator(object):

    MAX_LEN = 20
    PAD_TOKEN = '<|pad|>'
    STOP_TOKEN = '<|endoftext|>'

    def __init__(self):
        self.batch_size = int(args["--batch-size"])

        self.config = GPT2Config.from_pretrained("gpt2-medium")
      

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    

        model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.model = model.cuda()
        self.model.eval()

    def get_predictions_document(self, inpath, outpath):
        test_input = self.read_jsonl_to_list(inpath)

        """
        predictions = []
        num_batches = len(test_input) // self.batch_size
        print(f"Total batches = {num_batches}")
        
        with open(f"{outpath}", 'w') as out_file:
            for examples_batch in tqdm(self.batch_and_feed(test_input), total=num_batches):

                questions = [item["question"] for item in examples_batch]
                predictions = self.get_predictions_batch(questions)

                for i, prediction in enumerate(predictions):
                    examples_batch[i]["predicted_answer"] = prediction.strip()
                    json.dump(examples_batch[i], out_file)
                    out_file.write('\n')
        """
        with open(f"{outpath}", 'w') as out_file:
            for item in tqdm(test_input):
                question = item["question"]
                item["predicted_answer"] = self.get_predictions_sentence(
                    question).strip()
                json.dump(item, out_file)
                out_file.write('\n')

    def read_jsonl_to_list(self, pth: str) -> List[dict]:
        res = []
        with open(pth, 'r') as open_file:
            for line in open_file:
                res.append(json.loads(line))
        return res

    def get_predictions_sentence(self, input_sentence: str, beam_search: bool = False) -> str:
        encoded_prompt = self.tokenizer.encode(
            input_sentence, add_special_tokens=True, return_tensors='pt')
        encoded_prompt = encoded_prompt.cuda()
        prompt_length = encoded_prompt.shape[1]
        generation = ""
        with torch.no_grad():

            if beam_search:
                out = self.model.generate(
                    # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                    input_ids=encoded_prompt,
                    max_length=Gpt2Generator.MAX_LEN + prompt_length,
                    num_beams=5
                )
            else:
                out = self.model.generate(
                    # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                    input_ids=encoded_prompt,
                    max_length=Gpt2Generator.MAX_LEN + encoded_prompt.size(-1),
                    temperature=1.0,
                    top_k=0,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                )
            for out_seq in out:
                out_seq = out_seq[prompt_length:]
                text = self.tokenizer.decode(
                    out_seq, clean_up_tokenization_spaces=True)
                text = text[: text.find(
                    Gpt2Generator.STOP_TOKEN) if Gpt2Generator.STOP_TOKEN else None]
                #  answer = text[text.find('?')+1:]
                generation += text
                generation += '. '
            return generation

    def batch_and_feed(self, examples: List[dict]):
        """Creates iterators over examples
        Keyword Arguments:
            shf {bool} -- [Whether to input the shuffle] (default: {True})
        """
        size = len(examples)

        num_batches = size // self.batch_size
        if size % self.batch_size != 0:
            num_batches += 1
        for i in range(num_batches):
            examples_batch = examples[i *
                                      self.batch_size: (i + 1) * self.batch_size]
            yield examples_batch

    def get_predictions_batch(self, input_sentences: List[str]):

        encoded_prompts = []
        prompt_lengths = []
        max_length = max([len(sentence) for sentence in input_sentences])

        for sentence in input_sentences:
            tmp = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                             pad_to_max_length=True, max_length=max_length)
            encoded_prompts.append(tmp["input_ids"])
            prompt_lengths.append(tmp["attention_mask"].sum().item())

        encoded_prompts = torch.cat(encoded_prompts, dim=0).cuda()

        with torch.no_grad():
            out = self.model.generate(
                # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                input_ids=encoded_prompts,
                max_length=Gpt2Generator.MAX_LEN + max_length,
                temperature=1.0,
                top_k=0,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
            )

            decoded_answers = []

            for i, out_seq in enumerate(out):
                out_seq = out_seq[prompt_lengths[i]:]
                text = self.tokenizer.decode(
                    out_seq, clean_up_tokenization_spaces=True)
                text = text[: text.find(
                    Gpt2Generator.STOP_TOKEN) if Gpt2Generator.STOP_TOKEN else None]
                #  answer = text[text.find('?')+1:]
                decoded_answers.append(text + ". ")

            return decoded_answers


if __name__ == '__main__':
    args = docopt(__doc__)
    generator = Gpt2Generator()
    inpath = sys.argv[3]
    outpath = sys.argv[4]
    generator.get_predictions_document(
        inpath=args["--input-path"],
        outpath=args["--output-path"])
