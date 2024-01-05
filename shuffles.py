import os
import sys
import random
import itertools
from collections import deque

import rfutils
import rfutils.ordering
import unimorph
import numpy as np
import pandas as pd

import infoloc as il
import anipa_to_ipa

try:
    import cliqs.corpora
except ImportError:
    pass

PROTEINS_PATH = "/Users/canjo/data/genome/"
PROTEINS_FILENAME = "GRCh37_latest_protein.faa"
UNIMORPH_PATH = "/Users/canjo/data/unimorph/"
WOLEX_PATH = "/Users/canjo/data/wolex/"

def extract_manner(anipa_phone):
    if anipa_phone.startswith("C"):
        return anipa_phone[3]
    elif anipa_phone.startswith("V"):
        return "V"
    else:
        return anipa_phone # word or morpheme boundary

def extract_cv(anipa_phone):
    if anipa_phone:
        return anipa_phone[0]
    else:
        return ""

def sequence_transformer(f):
    def wrapped(s, *a, **k):
        restore = il.restorer(s)
        result = f(s, *a, **k)
        return restore(result)
    return wrapped

def delimited_sequence_transformer(f):
    def wrapped(s, *a, **k):
        restore = il.restorer(s)
        l = list(s)
        l2 = list(strip(s, il.DELIMITER))
        has_delimiter = l != l2
        r = f(l2, *a, **k)
        if has_delimiter:
            r = itertools.chain([il.DELIMITER], r, [il.DELIMITER])
            return restore(r)
        else:
            return restore(r)
    return wrapped

def add_delimiter_sequence(x):
    return type(x)(itertools.chain([il.DELIMITER], x, [il.DELIMITER]))

def shuffle_by_skeleton(xs, skeleton):
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
    return rfutils.ordering.reorder(xs, reordering)

def read_faa(filename):
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
                            'form': il.DELIMITER + "".join(so_far) + il.DELIMITER,
                        }
                    so_far.clear()
                    code, protein = line.strip(">").split(" ", 1)
                else:
                    so_far.append(line.strip())
    return pd.DataFrame(gen())

def read_wolex(filename):
    df = pd.read_csv(filename)
    df['form'] = df['word'].map(lambda x: tuple(anipa_to_ipa.segment(x)))
    df['ipa_form'] = df['word'].map(anipa_to_ipa.convert_word)
    return df

def reorder_manner(anipa_form):
    skeleton = list(map(extract_manner, anipa_form))
    return shuffle_by_skeleton(anipa_form, skeleton)

def reorder_cv(anipa_form):
    skeleton = list(map(extract_cv, anipa_form))
    return shuffle_by_skeleton(anipa_form, skeleton)

def read_unimorph(filename, with_delimiter='both'):
    with open(filename) as infile:
        lines = [line.strip().split("\t") for line in infile if line.strip()]
    lemmas, forms, features = zip(*lines)
    if not lines:
        raise FileNotFoundError
    result = pd.DataFrame({'lemma': lemmas, 'form': forms, 'features': features})
    result['lemma'] = result['lemma'].map(str.casefold)
    result['form'] = result['form'].map(str.casefold)    
    if with_delimiter == 'left':
        result['form'] = DELIMITER + result['form'] 
        result['lemma'] = DELIMITER + result['lemma']
    elif with_delimiter == 'both':
        result['form'] = DELIMITER + result['form'] + DELIMITER
        result['lemma'] = DELIMITER + result['lemma'] + DELIMITER
    return result

def parse_infl(s):
    return sorted(s.split("|"))

def read_ud(lang):
    def gen():
        for s in cliqs.corpora.ud_corpora[lang].sentences(fix_content_head=False):
            for n in s.nodes():
                if n != 0:
                    yield {
                        'word': s.node[n]['word'],
                        'lemma': s.node[n]['lemma'],
                        'pos': s.node[n]['pos'],
                        'infl': tuple(parse_infl(s.node[n]['infl'])),
                    }
    df = pd.DataFrame(gen())
    lemma_infl = []
    word_infl = []
    for infl, lemma, word in zip(df['infl'], df['lemma'], df['word']):
        lemma_infl.append(infl + ("Lemma="+lemma,))
        word_infl.append(infl + ("Word="+word,))
    df['word_infl'] = word_infl        
    df['lemma_infl'] = lemma_infl
    return df

def genome_comparison(**kwds):
    ds = DeterministicScramble(seed=0)
    scrambles = {'even_odd': even_odd, 'shuffled': ds.shuffle}
    return comparison(read_faa, PROTEINS_PATH, [PROTEINS_FILENAME], scrambles, **kwds)

def wolex_comparison(**kwds):
    wolex_filenames = [filename for filename in os.listdir(WOLEX_PATH) if filename.endswith(".Parsed.CSV-utf8")]
    ds = DeterministicScramble(seed=0)
    scrambles = {
        'even_odd': even_odd,
        'manner': reorder_manner,
        'shuffled': ds.shuffle,
        'cv': reorder_cv
    }
    return comparison(read_wolex, WOLEX_PATH, wolex_filenames, scrambles, **kwds)

def unimorph_comparison(**kwds):
    ds = DeterministicScramble(seed=0)
    scrambles = {'even_odd': even_odd, 'shuffled': ds.shuffle}
    return comparison(read_unimorph, UNIMORPH_PATH, unimorph.get_list_of_datasets(), scrambles, **kwds)


# (Mostly) agglutinative langs in UD:
# Uralic/Finnic: Finnish, Karelian, Livvi, Estonian
# Uralic/Mordvinic: Erzya, Moksha
# Uralic/Ugric: Hungarian
# Uralic/Permian: Komi
# Uralic/Sami: North Sami, Skolt Sami
# Turkic: Turkish, Kazakh, Uyghur
# East Asian: Japanese, Korean
def ud_morpheme_order_scores(lang, with_lemma=True):
    df = read_ud(lang)
    if with_lemma:
        data = df['lemma_infl']
    else:
        data = df['infl']    
    orders = list(morpheme_orders(data))
    random.shuffle(orders)
    for order in orders:
        yield total_order_score(il.ms_auc, data, order), total_order_score(il.ee_lower_bound, data, order), order
        
def comparison(read, path, langs, scrambles, maxlen=10, seed=0):
    for lang in langs:
        filename = os.path.join(path, lang)
        print("Analyzing", filename, file=sys.stderr)

        try:
            wordforms = read(filename)
        except FileNotFoundError:
            print("File not found", file=sys.stderr)
            continue
        n = len(wordforms)

        if 'count' in wordforms.columns:
            weights = wordforms['count']
        else:
            weights = None

        ht_real = il.curves_from_sequences(wordforms['form'], maxlen=maxlen, weights=weights)
        ht_real['real'] = 'real'
        ht_real['lang'] = lang
        ht_real['n'] = n
        yield ht_real

        for scramble_name, scramble_fn in scrambles.items():
            ht = il.curves_from_sequences(wordforms['form'].map(scramble_fn), maxlen=maxlen, weights=weights)
            ht['real'] = scramble_name
            ht['lang'] = lang
            ht['n'] = n
            yield ht
            
def strip(xs, y):
    result = xs
    if xs[0] == y:
        result = result[1:]
    if result[-1] == y:
        result = result[:-1]
    return result

class DeterministicScramble:
    def __init__(self, seed=0):
        self.seed = seed

    def shuffle(self, s):
        restore = il.restorer(s)
        r = list(strip(s, il.DELIMITER))
        np.random.RandomState(self.seed).shuffle(r)
        r.insert(0, il.DELIMITER)
        r.append(il.DELIMITER)
        return restore(r)

@delimited_sequence_transformer
def scramble_form(s):
    random.shuffle(s)
    return s

@delimited_sequence_transformer
def reorder_form(s, order):
    return rfutils.ordering.reorder(s, order)

@delimited_sequence_transformer
def reorder_total(s, total_order):
    def lookup_order(x):
        return total_order.index((x.split("=")[0],))
    return sorted(s, key=lookup_order)

@delimited_sequence_transformer
def fuse_morphemes(s, ordered_bundles):
    parts = dict(x.split("=") for x in s)
    for bundle in ordered_bundles:
        values = [parts.get(b, "") for b in bundle]
        if any(values):
            label = ",".join(bundle)
            value = ",".join(values)
            yield "=".join([label, value])

@delimited_sequence_transformer
def even_odd(s):
    return s[::2] + s[1::2]

@delimited_sequence_transformer
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

def extract_features(infl):
    return {fv.split("=")[0] for s in infl for fv in s}

def morpheme_orders(infl):
    features = list(extract_features(infl) - {'Lemma'})
    features = [(x,) for x in features]
    n = len(features)
    for order in itertools.permutations(range(n)):
        yield [('Lemma',)] + list(rfutils.ordering.reorder(features, order))

def order_score(J, f, data, weights=None):
    new_data = data.map(f)
    return il.score(J, new_data, weights)

def total_fusion_score(J, infl, fused_order, weights=None):
    return order_scores(J, lambda x: fuse_morphemes(x, fused_order), infl, weights=weights)

def total_order_score(J, infl, order, weights=None):
    return order_scores(J, lambda x: reorder_total(x, order), infl, weights=weights)

def permutation_scores(J, forms, weights, perms=None):
    # should take ~3hr to do a sweep over 9! permutations
    # of (3*3)!=362,880 permutations, 1296 are 3-3-contiguous (~3.6%)
    if perms is None:
        l = rfutils.the_only(forms.map(len).unique()) - 2 # 2 delimiters
        perms = itertools.permutations(range(l))
    for perm in perms:
        yield order_score(J, lambda s: reorder_form(s, perm), forms, weights), perm

def write_dfs(file, dfs):
    def gen():
        for df in dfs:
            for _, row in df.iterrows():
                yield dict(row)
    rfutils.write_dicts(file, gen())

def main(arg):
    if arg == 'unimorph':
        write_dfs(sys.stdout, unimorph_comparison(maxlen=10))
        return 0
    elif arg == 'wolex':
        write_dfs(sys.stdout, wolex_comparison(maxlen=10))
        return 0
    elif arg == 'genome':
        write_dfs(sys.stdout, genome_comparison(maxlen=10))
    else:
        print("Give me argument in {wolex, unimorph, genome}", file=sys.stderr)
        return 1
        
if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))


        
