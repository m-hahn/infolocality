import sys
import csv
import random
from collections import Counter, namedtuple

import tqdm
import conllu

ALLOW_MULTIPLE_ADJECTIVES = False
FIELD = 'lemma'
UNK = 'unknown'
FILENAMES = ["/Users/canjo/data/cliqs/ud-treebanks-v2.8/UD_English-GUM/all.conllu"]
NOUN, ADJ, NUM, DET = "N Adj Num Det".split()

NP = namedtuple('NP', [NOUN, DET, NUM, ADJ])

def one_if_any(xs):
    return xs[0] if xs else None

def extract(sentences, allow_multiple_adjectives=ALLOW_MULTIPLE_ADJECTIVES, field=FIELD):
    for s in sentences:
        for head in s:
            if head['upos'] == 'NOUN' and head[field] != UNK:
                np = {
                    NOUN: head[field],
                    ADJ: [],
                    NUM: [],
                    DET: [],
                }
                for dep in s:
                    if dep['head'] == head['id']:
                        if dep['upos'] == 'ADJ' and dep['deprel'] == 'amod':
                            np[ADJ].append(dep[field])
                        elif dep['upos'] == 'NUM' and dep['deprel'] == 'nummod':
                            np[NUM].append(dep[field])
                        elif dep['upos'] == 'DET' and dep['deprel'] == 'det':
                            np[DET].append(dep[field])
                            
                if not np[ADJ]:
                    adj = None
                elif allow_multiple_adjectives:
                    adj = " ".join(np[ADJ])
                else:
                    adj = random.choice(np[ADJ])
                    
                yield NP(np[NOUN], one_if_any(np[DET]), one_if_any(np[NUM]), adj)

def parse_files(filenames):
    for filename in filenames:
        with open(filename) as infile:
            yield from conllu.parse_incr(infile)

def extract_adjectives(*filenames):
    if not filenames:
        filenames = FILENAMES
    writer = csv.DictWriter(sys.stdout, fieldnames=[NOUN, ADJ])
    writer.writeheader()
    sentences = parse_files(filenames)
    nps = extract(tqdm.tqdm(sentences), allow_multiple_adjectives=True)
    for np in nps:
        d = {NOUN: np._asdict()[NOUN], ADJ: np._asdict()[ADJ]}
        writer.writerow(d)

def extract_nps(*filenames):
    if not filenames:
        filenames = FILENAMES
    writer = csv.DictWriter(sys.stdout, fieldnames=[NOUN, ADJ, NUM, DET])
    writer.writeheader()
    sentences = parse_files(filenames)
    nps = extract(tqdm.tqdm(sentences))
    for np in nps:
        writer.writerow(np._asdict())

if __name__ == '__main__':
    extract_nps(*sys.argv[1:])
                        
                
