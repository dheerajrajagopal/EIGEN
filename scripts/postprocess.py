import json
import math
import ast
import argparse

relations = ["helps", "hurts", "helped", "hurt"]
polarity = ["more", "less", "larger", "smaller", "fewer", "increase", "decrease", "increases", "decreases"]

def read_json(filename):
    with open(filename, "r") as inp:
        data = inp.read()
    data = data.split('\n')
    if data[-1] == '':
        data = data[:-1]
    return data

def split_and_write(data, args):
    
    n = math.ceil(len(data)/args.no_splits)
    s = 0
    for i in range(0, len(data), n):

        curr_data = data[i:i+n]
        s += 1
        with open(
            args.out_dir + 'test_split_' + str(s) + '.jsonl', 'w') as out:
            for j in range(len(curr_data)):
                out.write(curr_data[j] + '\n')

def write_file(data, key, dir, filename):
    with open(dir+filename, "w") as out:
        for datum in data:
            datum = ast.literal_eval(datum)
            out.write(datum[key] + '\n')

def remove_polarity(data, key):
    cut_data = []
    for datum in data:
        datum = ast.literal_eval(datum)
        toks = datum[key].split()
        
        flag = 0
        for wrd in polarity:
            if wrd in toks:
                flag = 1
                idx = toks.index(wrd)
                sent = toks[:idx] + toks[idx+1:]
                d = {"rem_pol_"+key: " ".join(sent)}
                cut_data.append(str(d))
                break
        if flag == 0:
            d = {"rem_pol_"+key: " ".join(toks)}
            cut_data.append(str(d))
    return cut_data 

def cut_sentence(data, key):
    cut_data = []
    for datum in data:
        datum = ast.literal_eval(datum)
        toks = datum[key].split()

        flag = 0
        for wrd in relations:
            if wrd in toks:
                flag = 1
                idx = toks.index(wrd)
                d = {"cut_"+key: " ".join(toks[idx+1:])}
                cut_data.append(str(d))
                break
        if flag == 0:
            d = {"cut_"+key: " ".join(toks)}
            cut_data.append(str(d))
    return cut_data

def write_jsonl(data, cut_answer, cut_pred, out_dir, filename):
    with open(out_dir + filename, "w") as out:
        for i in range(len(data)):
           orig = ast.literal_eval(data[i])
           cref = ast.literal_eval(cut_answer[i])
           cpre = ast.literal_eval(cut_pred[i])
           orig["cut_answer"] = cref["cut_answer"]
           orig["cut_predicted_answer"] = cpre["cut_predicted_answer"]
           out.write(str(orig) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str,
                         required=True, help="The input data dir.")
    parser.add_argument("--out_dir", default='', type=str,
                         help="The dir to save the output files.")
    parser.add_argument("--task_name", default='', type=str,
                         help="merge | split | sep | cut | add-cut | remove-polarity .")
    parser.add_argument("--no_splits", default=4, type=int,
                         help="No of files to split the file into.")
    args = parser.parse_args()

    if args.task_name == 'split':
        data = read_json(args.data_dir + 'test.jsonl')
        split_and_write(data, args)
    elif args.task_name == 'merge':
        data = []
        for i in range(1, args.no_splits+1):
            data += read_json(args.data_dir + 'pred_split_' + i + '.jsonl')
        write_file(data, 'answer', args.out_dir, 'reference.txt')
        write_file(data, 'predicted_answer', args.out_dir, 'predictions.txt')
    elif args.task_name == 'sep':
        data = read_json(args.data_dir + 'predictions.jsonl')
        write_file(data, 'answer', args.out_dir, 'reference.txt')
        write_file(data, 'predicted_answer', args.out_dir, 'predictions.txt')
    elif args.task_name == 'cut':
        data = read_json(args.data_dir + 'predictions.jsonl')
        cut_answer = cut_sentence(data, 'answer')
        cut_pred = cut_sentence(data, 'predicted_answer')
        write_file(cut_answer, 'cut_answer', args.out_dir, 'cut_reference.txt')
        write_file(cut_pred, 'cut_predicted_answer', args.out_dir, 'cut_predictions.txt')
    elif args.task_name == 'add-cut':
        data = read_json(args.data_dir + 'predictions.jsonl')
        cut_answer = cut_sentence(data, 'answer')
        cut_pred = cut_sentence(data, 'predicted_answer')
        write_jsonl(data, cut_answer, cut_pred, args.out_dir, 'all_predictions.jsonl')
    elif args.task_name == 'remove-polarity':
        data = read_json(args.data_dir + 'predictions.jsonl')
        cut_answer = cut_sentence(data, 'answer')
        cut_pred = cut_sentence(data, 'predicted_answer')
        rpol_ans = remove_polarity(cut_answer, 'cut_answer')
        rpol_pre = remove_polarity(cut_pred, 'cut_predicted_answer')
        write_file(rpol_ans, 'rem_pol_cut_answer', args.out_dir, 'rem_pol_cut_reference.txt')
        write_file(rpol_pre, 'rem_pol_cut_predicted_answer', args.out_dir, 'rem_pol_cut_predictions.txt')
    else:
        print('Wrong Task Name!')

if __name__ == "__main__":
    main()
