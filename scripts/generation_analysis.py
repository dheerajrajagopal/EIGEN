import nltk
import argparse
import ast

from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english')) 
relations = ["helps", "hurts", "helped", "hurt"]
gain_words = {"helps", "more", "higher", "increase", "stronger", "faster", "greater", "longer", "larger", "helping", "increases"}
loss_words = {"hurts", "less", "lower", "decrease", "weaker", "slower", "smaller", "hurting", "softer", "fewer", "decreases"}
all_ignore_words = gain_words.union(loss_words).union(stop_words)

def read_file(filename):
    with open(filename, "r") as inp:
        data = inp.read()
    data = data.split('\n')
    if data[-1] == '':
        data = data[:-1]
    return data

def overlap_strings(string1, string2):

    tok1 = nltk.word_tokenize(string1)
    tok2 = nltk.word_tokenize(string2)

    tok1 = [tok.lower() for tok in tok1]
    tok2 = [tok.lower() for tok in tok2]

    tok1 = [tok for tok in tok1 if tok.isalnum()]
    tok2 = [tok for tok in tok2 if tok.isalnum()]

    itok1 = set(tok1).difference(all_ignore_words)
    itok2 = set(tok2).difference(all_ignore_words)
    #print(itok1, itok2)

    return itok1.intersection(itok2), itok1, itok2, tok1, tok2

def overlap_prompt_ref(data):

    percent, ratio,  = 0, 0
    total_prompt, total_ref = 0, 0
    start = "In the context of "
    end = ", What does "
    len_start = len(start)
    for datum in data:
        datum = ast.literal_eval(datum)
        ref = datum["answer"]
        qut = datum["question"]
        idx1 = qut.index(start)
        idx2 = qut.index(end)
        prompt = qut[idx1+len_start:idx2]
        pro_ref, ig_len_ref, ig_len_prompt, len_ref, len_prompt = overlap_strings(ref, prompt)
        if len(pro_ref) > 0:
            percent += 1
        ratio += len(pro_ref)
        total_prompt += len(len_prompt)
        total_ref += len(len_ref)

    print("Percent Instances with Overlap: " + str(percent*100.0/len(data)))
    print("Ratio of overlap with prompt len: " + str(ratio*100.0/total_prompt))
    print("Ratio of overlap with ref len: " + str(ratio*100.0/total_ref))

def overlap_context(data):
    percent_ref, percent_pre= 0, 0
    ratio_ref, ratio_pre = 0, 0 
    total_ref, total_pre = 0, 0
    total_cont_ref, total_cont_pre = 0, 0
    count = 0
    for datum in data:
        datum = ast.literal_eval(datum)
        par = datum["context"]
        ref = datum["answer"]
        pre = datum["predicted_answer"]
        rat1, cont_ref, _, toks_ref, _ = overlap_strings(ref, par)
        rat2, cont_pre, _, toks_pre, _ = overlap_strings(pre, par)
        if len(rat1) > 0:
            percent_ref += 1
        if len(rat2) > 0:
            percent_pre += 1

        ratio_ref += len(rat1)
        ratio_pre += len(rat2)
        total_ref += len(toks_ref)
        total_pre += len(toks_pre)
        total_cont_ref += len(cont_ref)
        total_cont_pre += len(cont_pre)

        '''
        count += 1
        if count == 5:
            break
        '''

    print("Pcnt of instances with ref overlap: " + str(percent_ref*100.0/len(data)))
    print("Ratio of overlap with content length of ref: " + str(ratio_ref*100.0/total_cont_ref))
    print("Ratio of overlap with total length of ref: " + str(ratio_ref*100.0/total_ref))
    print('\n')
    print("Pcnt of instances with pred overlap: " + str(percent_pre*100.0/len(data)))
    print("Ratio of overlap with content length of pred: " + str(ratio_pre*100.0/total_cont_pre))
    print("Ratio of overlap with total length of pred: " + str(ratio_pre*100.0/total_pre))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str,
                         required=True, help="The input data dir.")
    parser.add_argument("--out_dir", default='', type=str,
                         help="The dir to save the output files.")
    parser.add_argument("--task_name", default='', type=str,
                         help="overlap | bleurt")
    args = parser.parse_args()

    if args.task_name == 'overlap':
        data = read_file(args.data_dir + 'predictions.jsonl')
        #overlap_prompt_ref(data)
        overlap_context(data)
    elif args.task_name == 'bleurt':
        data = read_file(args.data_dir)
        data = [float(datum) for datum in data]
        print('Avg Bleurt Score: ' + str(sum(data)/len(data)))

if __name__ == "__main__":
    main()