import json
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from torch.utils.data import DataLoader, SequentialSampler
from typing import List, Tuple

from src.data.data_reader import TWDataset
from src.util.utils import ensure_path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def generate(args, model, tokenizer, input_fp, output_fp):
    logger.info(f"Generating open state changes for input file: {input_fp} ...")
    ensure_path(output_fp)
    output_file = open(output_fp, 'w', encoding="UTF-8")
    gen_dataset = TWDataset(tokenizer=tokenizer, file_path=input_fp, skip_answer=True, block_size=args.block_size, cache_dir=None)
    args.gen_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    gen_sampler = SequentialSampler(gen_dataset)


    gen_dataloader = DataLoader(gen_dataset, sampler=gen_sampler, batch_size=args.gen_batch_size, collate_fn=None)
    logger.info(f"Num generations = {len(gen_dataset)} with batch size = {args.gen_batch_size}")
    model.eval()

    batch_num = 0
    num_batches = gen_dataloader.__len__()  # args.gen_batch_size

    for batch in tqdm(gen_dataloader, desc="Generating"):
        batch_num += 1
        percent_complete = 100 * (batch_num / num_batches)
        if percent_complete - int(percent_complete) == 0:
            print(f"#", end='')
        inputs, _, metadata = batch
        inputs = inputs.to(args.device)
        # inputs = inputs[0, :28].unsqueeze(0)
        # labels = labels.to(args.device)
        with torch.no_grad():
            # shape(out): batch_size * num_return_sequences, sequence_length
            #       i.e.: batch_size, sequence_length
            out_batchwise = model.generate(
                # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                input_ids=inputs,
                # max_len: answer_len plus the original ques_len
                max_length=args.length + inputs.size(-1),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                do_sample=True,
                pad_token_id=0,
                num_return_sequences=1,
            )
            for out, input_idx in zip(out_batchwise, metadata.tolist()):
                # clean up tokenization spaces => do not -> don't, " 're" => are etc.
                text = tokenizer.decode(out, clean_up_tokenization_spaces=True)

                # Remove the question context from the generated sequence.
                text = text[text.find('?')+1: text.find(args.stop_token) if args.stop_token else None]
                print(f"{out}")
                print(f"{text}")
                exit(0)
                output_dict = {'id': gen_dataset.get_original_id_for(item_idx=input_idx[0]), 'answer': text}
                json.dump(output_dict, output_file)
                output_file.write("\n")

    logger.info(f"Generation output written to : {output_fp}")
    output_file.close()
