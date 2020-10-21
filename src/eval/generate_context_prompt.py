"""Generates the context for the nodes.

Usage:
    generate_context.py [options]

Options:
    --model-path=<str>          Path to the model
    --qa-file-path=<str>        Path to the qa file
    --outpath=<str>             Path to the output file
    --graph-file-path=<str>     Path to the graph file
"""
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from src.data.creation.influence_graph import Rels
from src.data.creation.influence_graph import SourceTextFormatter
from src.data.creation.influence_graph import InfluenceGraph
import json
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
import sys
import re
from docopt import docopt
from typing import List, Tuple, Dict
sys.path.insert(0, '..')
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}


class ContextGenerator(object):

    MAX_LEN = 20

    STOP_TOKEN = '<|endoftext|>'
    PAD_TOKEN = '<|pad|>'

    def __init__(self, model_path, add_reverse=False):
        self.model_path = model_path
        #  self.config = GPT2Config.from_pretrained(self.model_path)
        with open(f"{self.model_path}/special_tokens_map.json", "r") as f:
            speical_tokens = json.load(f)
            #  special_tokens["pad_token"] = ContextGenerator.PAD_TOKEN

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        # self.tokenizer.add_special_tokens(speical_tokens)

        model = GPT2LMHeadModel.from_pretrained(self.model_path)

        self.relations = [Rels.special_tokens['helps'],
                          Rels.special_tokens['hurts']]

        if add_reverse:
            self.relations.extend([Rels.special_tokens['helped_by'],
                                   Rels.special_tokens['hurt_by']])

         #  The order of the relations above is important because the order in which context generator was fed the input is different
        self.context_relation_mapping = {
            Rels.special_tokens['helps']: "HELPS",
            Rels.special_tokens['hurts']: "HURTS",
            Rels.special_tokens['hurt_by']: "X-RELATION-HURTS",
            Rels.special_tokens['helped_by']: "X-RELATION-HELPS"
        }

        self.model = model.cuda()
        self.model.eval()

    def generate(self, qa_file_path: str, outpath: str, graph_id_to_prompt: Dict):
        test_input = self.read_jsonl_to_list(qa_file_path)

        def extract_cause_effect_from_ques(question: str) -> Tuple[str, str]:
            """Extracts cause and effect given questions
            'suppose the coal is carefully selected happens, how will it affect If less coal is broken down.'
            Arguments:
                question {[str]} -- [description]
            """
            #  question = question.lower()
            cause = re.search('suppose (.*)happens', question).group(1).strip()
            effect = re.search('how will it affect (.*)',
                               question).group(1).strip()
            return (cause, effect)

        predictions = []
        with open(outpath, 'w') as open_file:
            for item in tqdm(test_input):
                question = item["question"]["stem"]
                prompt = graph_id_to_prompt[item["metadata"]["graph_id"]]
                cause, effect = extract_cause_effect_from_ques(question)
                if len(cause) > 1 and cause[-1] == ".":
                    cause = cause[:-1]

                if len(effect) > 1 and effect[-1] == ".":
                    effect = effect[:-1]
                para_steps = " ".join(
                    [p.strip() for p in item["question"]["para_steps"] if len(p) > 0])

                item["question"]["generated_cause_context"] = self.generate_context_batched(
                    cause, para_steps, prompt)
                item["question"]["generated_effect_context"] = self.generate_context_batched(
                    effect, para_steps, prompt)
                json.dump(item, open_file)
                open_file.write('\n')

    def read_jsonl_to_list(self, pth: str) -> List[dict]:
        res = []
        with open(pth, 'r') as open_file:
            for line in open_file:
                res.append(json.loads(line))
        return res

    def generate_context_batched(self, node_text: str, para_text: str, prompt: str,
                                 add_reverse: bool = True,
                                 feed_individual_queries: bool = True) -> str:

        queries = self.create_queries(
            node_text=node_text, para_text=para_text,
            prompt=prompt,
            formatter=SourceTextFormatter.natural_question_formatter_with_prompt)
        context_str = ""
        if feed_individual_queries:
            contexts = []
            for query in queries:
                # a hack to avoid issues with different tokenizations
                contexts.append(self.batch_generate([query])[0])
        else:
            try:
                contexts = self.batch_generate(queries)
            except Exception as e:
                print(e)
                return context_str

        for i, rel in enumerate(self.relations):
            context_str += f" {self.context_relation_mapping[rel]} {contexts[i]}"

        return context_str.strip()

    def create_queries(self, node_text: str, para_text: str, prompt: str, formatter) -> List[str]:

        context = ""
        res = []
        for rel in self.relations:
            res.append(formatter(
                relation=rel, prompt=prompt, path_length=Rels.special_tokens['1'], src=node_text, para=para_text))
            # res.append(
            #    f"{rel} {Rels.special_tokens['1']} {Rels.special_tokens['node_sep']} {node_text} {Rels.special_tokens['para_sep']} {para_text}")
        return res

    def batch_generate(self, input_sentences: List[str]) -> str:
        encoded_prompts = []
        prompt_lengths = []
        for sent in input_sentences:
            encoded_prompts.append(self.tokenizer.encode(
                sent, add_special_tokens=True, return_tensors='pt'))

        # all prompts have the same number of tokens
        prompt_length = encoded_prompts[0].shape[1]
        encoded_prompts = torch.cat(encoded_prompts, dim=0)
        #encoded_prompts = self.tokenizer.encode(input_sentences, add_special_tokens=True, return_tensors='pt')
        encoded_prompts = encoded_prompts.cuda()
        outputs = []
        with torch.no_grad():
            out = self.model.generate(
                # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                input_ids=encoded_prompts,
                max_length=ContextGenerator.MAX_LEN + prompt_length,
                temperature=1.0,
                top_k=0,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
            )

            for i, out_seq in enumerate(out):
                out_seq = out_seq[prompt_length:]
                text = self.tokenizer.decode(
                    out_seq, clean_up_tokenization_spaces=True)

                text = text[: text.find(
                    ContextGenerator.STOP_TOKEN) if ContextGenerator.STOP_TOKEN else None]
                #  answer = text[text.find('?')+1:]
                if len(text) == 0:
                    outputs.append(".")
                else:
                    outputs.append(text + ".")

            return outputs


def read_graphs(graph_pth: str) -> Dict[str, InfluenceGraph]:
    """Reads influence graphs located on disk to a list of InfluenceGraph objects

    Arguments:
        graph_pth {str} -- [description]

    Returns:
        List[InfluenceGraph] -- [description]
    """
    graphs = dict()
    with open(graph_pth, "r") as f:
        for line in f:
            ig = InfluenceGraph(json.loads(line.strip()))
            graphs[ig.graph_id] = ig.prompt
    return graphs


if __name__ == '__main__':
    args = docopt(__doc__)
    context_generator = ContextGenerator(model_path=args["--model-path"])
    graph_id_to_prompt = read_graphs(args["--graph-file-path"])
    context_generator.generate(
        qa_file_path=args["--qa-file-path"], outpath=args["--outpath"], graph_id_to_prompt=graph_id_to_prompt)
