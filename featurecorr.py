import sys
import random
from collections import Counter

import conllu
import pandas as pd
import numpy as np
import scipy.stats
import tqdm
from plotnine import *

EN_CORPUS_FILENAMES = [
    "/Users/canjo/data/cliqs/ud-treebanks-v2.8/UD_English-GUM/all.conllu",
    "/Users/canjo/data/cliqs/ud-treebanks-v2.8/UD_English-GUMReddit/all.conllu",
    "/Users/canjo/data/cliqs/ud-treebanks-v2.8/UD_English-EWT/all.conllu",
]

DE_CORPUS_FILENAMES = [
    "/Users/canjo/data/cliqs/ud-treebanks-v2.8/UD_German-HDT/all.conllu",
    "/Users/canjo/data/cliqs/ud-treebanks-v2.8/UD_German-GSD/all.conllu",    
    "/Users/canjo/data/cliqs/ud-treebanks-v2.8/UD_German-LIT/all.conllu",    
]

CORPUS_FILENAMES = EN_CORPUS_FILENAMES
NORMS_FILENAME = "/Users/canjo/data/lancaster_norms/norms_binary.csv"
WORD_FIELD = 'Word'
TOKEN_FIELD = 'lemma'
MORPH_FEATURES = {'Number'}
TARGET_POS = {'NOUN'}
RELATIONS = {'obj'}
PAIR_POS = {('VERB', 'NOUN')}

def read_norms(norms_filename=NORMS_FILENAME,
               word_field=WORD_FIELD):
    norms = pd.read_csv(norms_filename)
    norms[word_field] = norms[word_field].map(str.casefold)
    norms.index = norms[word_field]
    vectors = norms.drop(word_field, axis=1).T
    return vectors

def load_corpus(corpus_filenames=CORPUS_FILENAMES):
    text = "\n".join(open(filename).read() for filename in corpus_filenames)
    corpus = conllu.parse(text)
    return corpus

def extract_corpus(vectors,
                   corpus,
                   target_features=MORPH_FEATURES,
                   target_pos=TARGET_POS):
    for sentence in corpus:
        for token in sentence:
            if ((cf_lemma := token['lemma'].casefold()) in vectors
                and token['upos'] in target_pos
                and token['feats']
                and any(feature in token['feats'] for feature in target_features)):
                features = dict(vectors[cf_lemma])
                for feature in target_features:
                    features[feature] = token['feats'].get(feature)
                yield tuple(features.values())

def extract_word_pairs(corpus,
                       relations=RELATIONS,
                       target_pos=PAIR_POS,
                       corpus_filenames=CORPUS_FILENAMES,
                       field=TOKEN_FIELD):
    for sentence in corpus:
        for token in sentence:
            if token['deprel'] in relations:
                if (head_pos := token['head']) > 0:
                    head = sentence[head_pos - 1]
                    if ((head['upos'], token['upos']) in target_pos
                        and (headf := head[field]).isalpha()
                        and (depf := token[field]).isalpha()):
                        yield headf, depf

def mi_from_pair_counts(keys, counts):
    counts = list(counts)
    x = Counter()
    y = Counter()
    Z = 0
    for (one, two), count in zip(keys, counts):
        x[one] += count
        y[two] += count
        Z += 1
    return (
        scipy.stats.entropy(list(x.values())) +
        scipy.stats.entropy(list(y.values())) -
        scipy.stats.entropy(counts)
    )
                        
def pairwise_mis(feats, names, field='count'):
    print("Calculating MIs...", file=sys.stderr)
    k = len(names)
    def mi(i, j):
        marginal1 = feats[[i,field]].groupby([i]).sum().reset_index()
        marginal2 = feats[[j,field]].groupby([j]).sum().reset_index()
        joint = feats[[i,j,field]].groupby([i,j]).sum().reset_index()        
        return (
            scipy.stats.entropy(marginal1[field]) +
            scipy.stats.entropy(marginal2[field]) -
            scipy.stats.entropy(joint[field])
        )

    def pairwise():
        for i in tqdm.tqdm(range(k-1)):
            for j in range(i+1,k):
                yield names[i], names[j], mi(i, j)

    result = pd.DataFrame(pairwise())
    result.columns = ['i', 'j', 'mi']
    return result

def word_pair_mis(corpus_filenames=CORPUS_FILENAMES,
                  norms_filename=NORMS_FILENAME,
                  word_field=WORD_FIELD,
                  relations=RELATIONS,
                  target_pos=PAIR_POS,
                  token_field=TOKEN_FIELD):
    vectors = read_norms(norms_filename=norms_filename, word_field=word_field)
    names = (
        ["h_"+name for name in vectors.index] +
        ["d_"+name for name in vectors.index]
    )
    corpus = load_corpus(corpus_filenames)
    tokens = extract_word_pairs(
        tqdm.tqdm(corpus),
        relations=relations,
        target_pos=target_pos,
        field=token_field,
    )
    vector_counts = Counter(
        tuple(vectors[head_lemma]) + tuple(vectors[dep_lemma])
        for head, dep in tokens
        if (head_lemma := head.casefold()) in vectors and (dep_lemma := dep.casefold()) in vectors
    )
    feats = pd.DataFrame(vector_counts.keys())
    feats['count'] = vector_counts.values()
    df = pairwise_mis(feats, names)
    df['within'] = (
        (df['i'].map(lambda s: s.startswith('d_')) & df['j'].map(lambda s: s.startswith('d_'))) |
        (df['i'].map(lambda s: s.startswith('h_')) & df['j'].map(lambda s: s.startswith('h_'))) 
    )
    return df

def shuffled(xs):
    ys = list(xs)
    random.shuffle(ys)
    return ys

def raw_word_pair_counts(corpus_filenames=CORPUS_FILENAMES,
                         target_pos=PAIR_POS,
                         relations=RELATIONS,
                         token_field=TOKEN_FIELD):
    corpus = load_corpus(corpus_filenames)
    pairs = extract_word_pairs(
        tqdm.tqdm(corpus),
        relations=relations,
        target_pos=target_pos,
        field=token_field,
    )
    counts = Counter(pairs)
    return counts

def word_feature_mis(corpus_filenames=CORPUS_FILENAMES,
                     norms_filename=NORMS_FILENAME,
                     word_field=WORD_FIELD,
                     target_features=MORPH_FEATURES,
                     target_pos=TARGET_POS):

    vectors = read_norms(norms_filename=norms_filename, word_field=word_field)
    names = list(vectors.index) + list(target_features)
    corpus = load_corpus(corpus_filenames)

    counts = Counter(extract_corpus(
        vectors,
        tqdm.tqdm(corpus),
        target_features=target_features,
        target_pos=target_pos
    ))

    feats = pd.DataFrame(counts.keys())
    feats['count'] = counts.values()    
    df = pairwise_mis(feats, names)
    df['within'] = ~((df['i'].map(lambda x: x in target_features)) | (df['j'].map(lambda x: x in target_features)))
    return df

def plot(df):
    df = df.sort_values('mi')
    df['rank'] = range(len(df))
    df['label'] = df['i'].map(lambda s: s.strip('.mean')) + " " + df['j'].map(lambda s: s.strip('.mean'))
    return ggplot(df, aes(x='rank', y='mi', fill='within', color='within', label='label')) + geom_bar(stat="identity") + theme_classic() + geom_text(angle=90)    

    
def main(type='morpheme'):
    if type.startswith('morpheme'):
        df = word_feature_mis()
    elif type.startswith('word'):
        df = word_pair_mis()
    df.to_csv(sys.stdout)

if __name__ == '__main__':
    main(*sys.argv[1:])

