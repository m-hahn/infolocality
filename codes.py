from typing import *
import itertools
import random

import numpy as np
import scipy.special
import pandas as pd

import infoloc as il
import huffman

DELIMITER = '#'

def segments(iterable, breakpoints):
    """ Segments

    Break iterable into contiguous segments at specified index
    breakpoints. New iterators are formed starting with each breakpoint.

    Params:
        iterable: An iterable to be broken into segments.
        breakpoints: An iterable of integers representing indices
            for where to break the iterable.

    Yields:
        Sequences of items from iterable.
    """
    iterable = list(iterable)
    breakpoints = sorted(set(breakpoints))
    start = 0
    for breakpoint in breakpoints:
        yield iterable[start:breakpoint]
        start = breakpoint
    yield iterable[start:]

def ljust(xs, length, value):
    xs = list(xs)
    num_needed = length - len(xs)
    if num_needed <= 0:
        return xs
    else:
        xs.extend([value]*num_needed)
        return xs

def int_to_char(x, offset=65):
    return chr(offset + x)

def ints_to_str(ints, offset=65):
    return "".join(map(int_to_char, ints))

# How many "effective phonemes" are there in a language?
# This is given by e^h.
# For example for English orthography (unimorph), h = 1.148 and H = 2.9538
# so the effective grapheme inventory given contextual redundancy is 3.15,
# and the effective grapheme inventory given unigram grapheme frequencies is 19.18
# So English writing could get away with using only 4 symbols.
# Stenographers' keyboards have ~20... similar to the unigram perplexity.
# 

def huffman_lexicon(forms, weights, n, with_delimiter=True):
    codebook = huffman.huffman(weights, n=n)
    if with_delimiter == 'left':
        return [DELIMITER + ints_to_str(code) for code in codebook]
    elif with_delimiter:
        return [DELIMITER + ints_to_str(code) + DELIMITER for code in codebook]
    else:
        return list(map(ints_to_str, codebook))

def rand_str(V, k):
    ints = [random.choice(range(V)) for _ in range(k)]
    return ints_to_str(ints)

def all_same(xs):
    x, *rest = xs
    return all(r == x for r in rest)

def is_contiguous(k, l, perm):
    """ Return whether permutation perm when applied to a string with k components of length l
    preserves contiguity and whether it preserves the order of elements within each component consistently """
    canonical_order = range(k*l)
    breaks = [l*k_ for k_ in range(1,k)]
    words = segments(canonical_order, breaks)
    index_sets = {frozenset(word) for word in words}
    contiguous = all(
        frozenset(perm_segment) in index_sets
        for perm_segment in segments(perm, breaks)
    )
    invariant_words = [
        tuple(i-min(i for i in word) for i in word)
        for k, word in enumerate(segments(perm, breaks))
    ]
    consistent = all_same(invariant_words)
    return contiguous, consistent

def test_is_contiguous():
    assert all(is_contiguous(2, 2, (0,1,2,3)))
    assert is_contiguous(2, 2, (1,0,2,3))[0]
    assert not is_contiguous(2, 2, (1,0,2,3))[1]
    assert all(is_contiguous(2, 3, (2,1,0,5,4,3)))
    assert is_contiguous(2, 3, (2,1,0,5,3,4))[0]
    assert not is_contiguous(2, 3, (2,1,0,5,3,4))[1]
    assert all(is_contiguous(3, 3, (2,1,0,5,4,3,8,7,6)))
    assert all(is_contiguous(2, 2, (3,2,1,0)))
    assert all(is_contiguous(2, 4, (7, 6, 5, 4, 3, 2, 1, 0)))

def encode_contiguous_positional_random_order(ms, code):
    """ Use code to encode each element of ms in a random order. """
    vocab_size = code.max() + 1
    signals = [code[m]+vocab_size*k for k, m in enumerate(ms)]
    np.random.shuffle(signals)
    return np.hstack(signals)

def encode_contiguous_positional(ms, code):
    vocab_size = code.max()+1
    return np.hstack([code[m]+vocab_size*k for k, m in enumerate(ms)])

def encode_contiguous(ms, code):
    """ Use code to encode each element of ms, then concatenate them. """
    return np.hstack(code[ms,:])

def encode_weak_contiguous(ms, codes):
    """ Use codes to encode each element of ms. """
    # ms is a sequence of morpheme ids
    return np.hstack([code[m] for m, code in zip(ms, codes)])

def word_probabilities(p_Mk, code, encode=encode_contiguous, with_delimiter=True):
    def gen():
        for i, mk in enumerate(itertools.product(*map(range, p_Mk.shape))):
            yield ints_to_str(encode(mk, code)), p_Mk[mk]
    df = pd.DataFrame(gen())
    df.columns = ['form', 'probability']
    if with_delimiter == 'left':
        df['form'] = DELIMITER + df['form']
    elif with_delimiter:
        df['form'] = DELIMITER + df['form'] + DELIMITER
    return df.groupby(['form']).sum().reset_index()

def as_code(code: np.array):
    def f(m):
        return "".join(map(str, code[m]))
    return f

concatenate = "".join

def char_gensym(x, _state={}):
    if x in _state:
        return _state[x]
    else:
        _state[x] = int_to_char(len(_state))
        return _state[x]

def identity_code(features):
    return "".join(map(char_gensym, features))

def systematic_code(code, combination_fn=concatenate):
    def composed_code(features: Iterable) -> str: 
        strings = map(code, features)
        return combination_fn(strings)
    return composed_code

def weakly_systematic_code(codes, combination_fn=concatenate):
    def composed_code(features: Iterable) -> str:
        return combination_fn(code(feature) for code, feature in zip(codes, features))
    return composed_code

def random_systematic_code(meanings, S, l, unique=False, combination_fn=concatenate):
    # meanings is an iterable of feature bundles. Feature bundles are iterables of features.
    # strongly systematic.
    value_set = {feature for feature_bundle in meanings for feature in feature_bundle}
    random_digits = random_code(len(value_set), S, l, unique=unique)
    codebook = dict(zip(value_set, map(ints_to_str, random_digits)))
    return systematic_code(codebook.__getitem__, combination_fn=combination_fn), codebook

def form_probabilities(p, meanings, code, with_delimiter='both'):
    """ code is a mapping from meanings (iterables of feature bundles) to strings """
    forms = map(code, meanings)
    df = pd.DataFrame({'form': forms, 'probability': p})
    if with_delimiter == 'left':
        df['form'] = DELIMITER + df['form']
    elif with_delimiter:
        df['form'] = DELIMITER + df['form'] + DELIMITER    
    return df.groupby(['form']).sum().reset_index()

def form_probabilities_np(source, code, with_delimiter='both'):
    """ code is an array of same-length integers representing symbols """
    def gen():
        for i, m in enumerate(itertools.product(*map(range, source.shape))):
            yield ints_to_str(m), ints_to_str(code[m]), source[m]
    df = pd.DataFrame(gen())
    df.columns = ['meaning', 'form', 'probability']
    if with_delimiter == 'left':
        df['form'] = DELIMITER + df['form']
    elif with_delimiter:
        df['form'] = DELIMITER + df['form'] + DELIMITER
    return df[['form', 'probability']].groupby(['form']).sum().reset_index()

def random_code(M, S, l, unique=False):
    """ 
    Input:
    M: Number of messages.
    S: Number of distinct signals/forms.
    l: Signal length.

    Output:
    An M x S^l array e, where e[m,:] = the length-l code for m.
    """
    if unique:
        codes = set()
        while len(codes) < M:
            proposed = tuple(np.random.randint(S, size=l))
            if proposed in codes:
                continue
            else:
                codes.add(proposed)
        return np.array(list(codes))
    else:
        return np.random.randint(S, size=(M, l))

def paradigms(num_meanings, num_words):
    def relabel(sequence):
        state = itertools.count()
        mapping = {}
        for x in sequence:
            if x not in mapping:
                mapping[x] = next(state)
                yield mapping[x]
            else:
                yield mapping[x]
    sequences = set()                
    for sequence in cartesian_indices(num_words, num_meanings):
        relabeled = tuple(relabel(sequence))
        if max(relabeled) == num_words - 1:
            sequences.add(relabeled)
    return sequences

def uniform_code(M, S):
    uniform_code_len = np.log(N) / np.log(num_signals) # what is N?
    uniform_code = cartesian_indices(num_signals, int(np.ceil(uniform_code_len)))
    return BROKEN#np.array(list(take(uniform_code, N)))

def cartesian_power(xs, k):
    xs = list(xs)
    return itertools.product(*[xs]*k)

def cartesian_indices(V, k):
    return itertools.product(*[range(V)]*k)

def cartesian_distinct_indices(V, k):
    for sequence in cartesian_indices(V, k):
        yield tuple(i*V + x for i, x in enumerate(sequence))

def cartesian_distinct_forms(V, k):
    numerals = cartesian_indices(V, k)
    for group in numerals:
        yield "".join(int_to_char(x+i*V) for i, x in enumerate(group))

def cartesian_forms(V, k):
    numerals = cartesian_indices(V, k)
    for group in numerals:
        yield "".join(map(int_to_char, group))

flat = itertools.chain.from_iterable        

def repeating_blocks(V, k, m, overlapping=True):
    vocab = list(range(V))
    def gen():
        for vs in cartesian_indices(V, m):
            parts = [
                [(1-overlapping)*b*V + vs[b]]*k
                for b in range(m)
            ]
            yield DELIMITER + "".join(flat(ints_to_str(x) for x in parts)) + DELIMITER
    return pd.DataFrame({'form': list(gen())})

if __name__ == '__main__':
    import nose
    nose.runmodule()
