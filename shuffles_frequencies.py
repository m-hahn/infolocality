import os
import sys
import random
import itertools
from collections import deque
from typing import *
import collections, string

import tqdm
import numpy as np
import pandas as pd

import utils
import infoloc as il
import anipa_to_ipa

DEFAULT_DELIMITER = utils.RightDelimiter()
DEFAULT_COUNT_DELIMITER = "\t"

PROTEINS_PATH = "/Users/canjo/data/genome/GRCh37_latest_protein.faa"
WOLEX_PATH = "/Users/michaelhahn/Downloads/wolex/original/"

DEFAULT_NUM_SAMPLES = 10

T = TypeVar("T", bound=Any)

def reorder(xs: Iterable[T], indices: Iterable[int]) -> Sequence[T]:
    """ reorder

    Elements of xs in the order specified by indices.
    For all i, reorder(xs, order)[i] == xs[order[i]].
    
    Example:
    >>> list(reorder(['a', 'b', 'c', 'd'], [3, 1, 2, 0]))
    ['d', 'b', 'c', 'a']

    """
    xs = list(xs)
    indices = list(indices)
    assert len(xs) == len(indices)
    return [xs[i] for i in indices]

def shuffle_preserving_length(forms: pd.Series, granularity: int=1) -> pd.Series:
    lenclass = forms.map(len) // granularity
    assert utils.is_monotonically_increasing(lenclass)
    new_forms = []
    for length in lenclass.drop_duplicates(): # ascending order
        mask = lenclass == length
        shuffled_forms = np.random.permutation(forms[mask])
        new_forms.extend(shuffled_forms)
    return new_forms

def extract_manner(anipa_phone: str) -> str:
    if anipa_phone.startswith("C"):
        return anipa_phone[3]
    elif anipa_phone.startswith("V"):
        return "V"
    else:
        return anipa_phone # word or morpheme boundary

def extract_cv(anipa_phone: str) -> str:
    if anipa_phone:
        return anipa_phone[0]
    else:
        return ""

def shuffle_by_skeleton(xs: Sequence[T], skeleton: Sequence[str]) -> Sequence[T]:
    """ Shuffle xs while retaining the invariant described by skeleton. """
    # The skeleton is assumed to contain any delimiters
    assert len(skeleton) == len(xs)
    # For example, xs = "static", skeleton = "stVtVt"
    reordering = [None] * len(xs)
    for y in set(skeleton): # iterating through {t, s, V}
        old_indices = [i for i, y_ in enumerate(skeleton) if y_ == y] # first pass, [1, 3, 5]
        new_indices = old_indices.copy()
        random.shuffle(new_indices) # first pass, [3, 5, 1]
        for old_index, new_index in zip(old_indices, new_indices):
            reordering[old_index] = new_index # first pass, [None, 3, None, 5, None, 1]
    assert all(i is not None for i in reordering)
    return reorder(xs, reordering)

def read_faa(filename: str, with_delimiter: utils.Delimiter=DEFAULT_DELIMITER) -> pd.DataFrame:
    def gen():
        so_far = []
        code = None
        protein = None
        with open(filename) as infile:
            for line in infile:
                if line.startswith(">"):
                    if code is not None:
                        yield {
                            'protein': protein,
                            'code': code,
                            'form': with_delimiter.delimit_string("".join(so_far)),
                        }
                    so_far.clear()
                    code, protein = line.strip(">").split(" ", 1)
                else:
                    so_far.append(line.strip())
    return pd.DataFrame(gen())

def read_wolex(filename: str, with_delimiter: utils.Delimiter=DEFAULT_DELIMITER) -> pd.DataFrame:
    df = pd.read_csv(filename, encoding='utf-8')
    print("Finished reading from the file", file=sys.stderr)
    df['form'] = df['word'].map(lambda x: tuple(anipa_to_ipa.segment(x))).map(lambda xs: utils.strip(xs, '#')).map(with_delimiter.delimit_sequence)
    print("Created form", file=sys.stderr)
    df['ipa_form'] = with_delimiter.delimit_array(df['word'].map(lambda s: s.strip("#")).map(anipa_to_ipa.convert_word))
    print("Created IPA form", file=sys.stderr)
    return df

def genome_comparison(**kwds):
    df = read_faa(PROTEINS_PATH)
    return comparison(df, baselines=['ds'], **kwds)


import re

def contains_non_alphabetic(x):
    # Regular expression pattern to match any character not in a-z or A-Z
    pattern = r'[^a-zA-Z]'
    
    # Search for the pattern in the string
    if re.search(pattern, x):
        return True
    else:
        return False


def convert_german_chars(s):
    replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss'
    }
    for char, replacement in replacements.items():
        s = s.replace(char, replacement)
    return s



def wolex_comparison(**kwds):
    print(kwds, file=sys.stderr)
    wolex_filenames = [
        filename for filename in os.listdir(WOLEX_PATH)
        if filename.endswith(".Parsed.CSV") and filename.split(".")[0] in kwds["languages"]
    ]
    print(os.listdir(WOLEX_PATH), file=sys.stderr)
    for filename in wolex_filenames:
        print(filename, file=sys.stderr)
        wolex = read_wolex(os.path.join(WOLEX_PATH, filename))
        print("finished reading Wolex", file=sys.stderr)
        language = filename.split(".")[0]
        if language == "SouthernBritishEnglish":
            language = "English"
        with open("/Users/michaelhahn/Downloads/wolex/original/VOCAB_FOR_WOLEX_FULL3/"+language.lower()+"-vocab_ALL.txt", "r") as inFile:
           wordCounts = dict([(x[0], int(x[1])) for x in [x.split("\t") for x in inFile.read().strip().split("\n")]])
        wordCounts_normalized = collections.defaultdict(int)
        weights = []
        for x, c in wordCounts.items():
            x_ = ''.join(c for c in x.lower() if c not in string.punctuation)
            if language == "German":
               x_ = convert_german_chars(x_)
            wordCounts_normalized[x_] += c
        j = 0
        for x in wolex['Orthography']:
            j += 1
#            print("orthographic form", x, file=sys.stderr)
            if x != x:
                print("WARNING! NA FORM", file=sys.stderr)
                weights.append(0)
            else:
                weights.append(wordCounts_normalized[x.lower()]+1)
      #          if contains_non_alphabetic(x):
       #           print(f"WARNING###{x}###", file=sys.stderr)
            print(j, x, weights[-1], file=sys.stderr)
                
        wolex['count'] = weights



        yield from comparison(wolex, baselines=['ds', 'manner', 'cv'], label=filename, **kwds)
            
class DeterministicScramble:
    def __init__(self, seed: int=0, with_delimiter: utils.Delimiter=DEFAULT_DELIMITER):
        self.seed = seed
        self.shuffle = utils.delimited_sequence_transformer(self.scramble)

    def scramble(self, s: Sequence[T]) -> Sequence[T]:
        np.random.RandomState(self.seed).shuffle(s)
        return s

reorder_form = utils.delimited_sequence_transformer(reorder)

@utils.delimited_sequence_transformer
def even_odd(s):
    return s[::2] + s[1::2]

@utils.delimited_sequence_transformer
def outward_in(s):
    s = deque(s)
    r = []
    while True:
        if s:
            first = s.popleft()
            r.append(first)
        else:
            break
        if s:
            last = s.pop()
            r.append(last)
        else:
            break
    return r

@utils.delimited_sequence_transformer
def reorder_manner(anipa_form):
    skeleton = list(map(extract_manner, anipa_form))
    return shuffle_by_skeleton(anipa_form, skeleton)

@utils.delimited_sequence_transformer
def reorder_cv(anipa_form):
    skeleton = list(map(extract_cv, anipa_form))
    return shuffle_by_skeleton(anipa_form, skeleton)

@utils.delimited_sequence_transformer
def reorder_total(form: Sequence, order: Sequence) -> Sequence:
    return sorted(form, key=order.index)

SIMPLE_BASELINES = {
    'manner': reorder_manner,
    'cv': reorder_cv,
    'even_odd': even_odd,
    'outward_in': outward_in,
}

DEFAULT_BASELINES = ['nonsys', 'nonsysl', 'ds']
ALLOWED_BASELINES = {'nonsys', 'nonsyl', 'ds', 'manner', 'cv', 'even_odd', 'outward_in', 'permutations'}

def default_comparison(
        filename: str,
        baselines: Optional[Sequence[str]]=DEFAULT_BASELINES,
        count_delimiter: str=DEFAULT_COUNT_DELIMITER,
        form_delimiter: Optional[str]=None,
        maxlen: Optional[int]=None,
        header: bool=False,
        num_samples: int=DEFAULT_NUM_SAMPLES) -> Iterator[pd.DataFrame]:

    df = pd.read_csv(filename, header=0 if header else None, delimiter=count_delimiter)
    if len(df.columns) == 1:
        df.columns = ['form']
    else:
        df.columns = ['form', 'count']
    if form_delimiter is not None:
        df['form'] = df['form'].map(lambda x: tuple(x.split(form_delimiter)))
    return comparison(df, baselines=baselines, maxlen=maxlen, num_samples=num_samples)

def comparison(
        df: pd.DataFrame,
        baselines: Optional[Sequence[str]]=DEFAULT_BASELINES,
        maxlen: int=None,
        seed: int=0,
        label: Optional[str]=None,
        num_samples: int=DEFAULT_NUM_SAMPLES,
        monitor: bool=True, **kwds) -> Iterator[pd.DataFrame]:
   
#        print(x, wordCounts_normalized[x.lower()])
#    print(wordCounts)
#    quit()    
 
    if 'count' in df.columns:
        weights = df['count']
    else:
        weights = None
        assert False

    if maxlen is None:
        maxlen = df['form'].map(len).max()

    kwds = {
        'maxlen': maxlen,
        'weights': weights,
        'monitor': monitor,
    }

    ht_real = il.curves_from_sequences(df['form'], **kwds)
    ht_real['real'] = 'real'
    ht_real['label'] = label
    ht_real['sample'] = 0
    yield ht_real

    if 'permutations' in baselines:
        perms = itertools.permutations(range(maxlen))
        for i, perm in enumerate(perms):
            ht = il.curves_from_sequences(df['form'].map(lambda x: reorder_form(x, perm)), **kwds)
            ht['real'] = 'perm_%d' % i
            ht['label'] = label
            ht['sample'] = 0
            yield ht

    for i in tqdm.tqdm(range(num_samples)):
        if 'nonsys' in baselines:
            print("Running nonsys baseline", file=sys.stderr)
            if weights is None:
                print("Warning: nonsys baseline is useless without counts.", file=sys.stderr)
            ht = il.curves_from_sequences(utils.shuffled(df['form']), **kwds)
            ht['real'] = 'nonsys'
            ht['label'] = label
            ht['sample'] = i
            yield ht


        if 'nonsysl' in baselines:
            print("Running nonsysl baseline", file=sys.stderr)            
            if weights is None:
                print("Warning: nonsysl baseline is useless without counts.", file=sys.stderr)            
            ht = il.curves_from_sequences(shuffle_preserving_length(df['form']), **kwds)
            ht['real'] = 'nonsysl'
            ht['label'] = label
            ht['sample'] = i
            yield ht

        if 'ds' in baselines:
            print("Running ds baseline", file=sys.stderr)            
            ht = il.curves_from_sequences(df['form'].map(DeterministicScramble(seed+i).shuffle), **kwds)
            ht['real'] = 'scramble'
            ht['label'] = label
            ht['sample'] = i
            yield ht

        for baseline in baselines:
            if baseline in SIMPLE_BASELINES:
                print("Running %s baseline" % baseline, file=sys.stderr)
                ht = il.curves_from_sequences(df['form'].map(SIMPLE_BASELINES[baseline]), **kwds)
                ht['real'] = baseline
                ht['label'] = label
                ht['sample'] = i
                yield ht

def main(args) -> int:
    if args.filename == 'wolex':
        # Generate data for Figure 3A
        utils.write_dfs(sys.stdout, wolex_comparison(
            maxlen=args.maxlen,
            num_samples=args.num_samples,
            languages=["SouthernBritishEnglish", "Dutch", "German", "French"],
        ))
        return 0
    elif args.filename == 'genome':
        utils.write_dfs(sys.stdout, genome_comparison(
            maxlen=args.maxlen, num_samples=args.num_samples
        ))
        return 0
    else:
        baselines = args.baselines.split(" ")
        assert all(baseline in ALLOWED_BASELINES for baseline in baselines)
        comparison = default_comparison(
            args.filename,
            baselines=baselines,
            count_delimiter=args.count_delimiter,
            form_delimiter=args.form_delimiter,
            maxlen=args.maxlen,
            num_samples=args.num_samples,
            header=args.header,
        )
        utils.write_dfs(sys.stdout, comparison)
        return 0
        
if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser("Estimate entropy rate curves from a file and compare to baselines. If argument is 'wolex', runs on all Wolex corpora. If argument is 'genome', runs on genome data. Otherwise argument is treated as a filename, for a file assumed to consist of at least two columns. The first column is treated as forms. The second column is treated as counts.")
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-m', '--maxlen', type=int, default=None)
    argparser.add_argument('-f', '--form_delimiter', type=str, default=None)
    argparser.add_argument('-c', '--count_delimiter', type=str, default=DEFAULT_COUNT_DELIMITER)
    argparser.add_argument('-n', '--num_samples', type=int, default=DEFAULT_NUM_SAMPLES)
    argparser.add_argument('-b', '--baselines', type=str, default=None)
    argparser.add_argument('--header', action='store_true', default=False)    
    args = argparser.parse_args()
    sys.exit(main(args))


        
