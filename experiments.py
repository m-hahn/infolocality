import itertools
import functools
import random
from collections import Counter

import rfutils
import tqdm
import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
from plotnine import *

import sources as s
import codes as c
import infoloc as il

def fusion_advantage(mi=1/2):
    p = s.mi_mix(mi)
    agglutinative = list(c.cartesian_distinct_forms(2, 2))
    fusional = agglutinative.copy()
    fusional[-1], fusional[-2] = fusional[-2], fusional[-1]
    agglutinative_curve = il.curves_from_sequences(agglutinative, weights=p)
    fusional_curve = il.curves_from_sequences(fusional, weights=p)
    agglutinative_curve['lang'] = 'agglutinative'
    fusional_curve['lang'] = 'fusional'
    result = pd.concat([agglutinative_curve, fusional_curve])
    result['mi'] = mi
    return result

def agglutination_advantage(mi=1/2):
    p = s.product_distro(s.flip(1/2), s.mi_mix(mi))
    agglutinative = list(c.cartesian_distinct_forms(2, 3))
    fusionalgood = agglutinative.copy()
    fusionalgood[2], fusionalgood[3] = fusionalgood[3], fusionalgood[2]
    fusionalgood[6], fusionalgood[7] = fusionalgood[7], fusionalgood[6]

    fusionalbad = agglutinative.copy()
    (fusionalbad[4], fusionalbad[5]), (fusionalbad[6], fusionalbad[7]) = (fusionalbad[6], fusionalbad[7]), (fusionalbad[4], fusionalbad[5])
    agglutinative_curve = il.curves_from_sequences(agglutinative, weights=p)
    fusionalbad_curve = il.curves_from_sequences(fusionalbad, weights=p)
    fusionalgood_curve = il.curves_from_sequences(fusionalgood, weights=p)
    agglutinative_curve['lang'] = 'agglutinative'
    fusionalbad_curve['lang'] = 'fusional_bad'
    fusionalgood_curve['lang'] = 'fusional_good'
    result = pd.concat([agglutinative_curve, fusionalbad_curve, fusionalgood_curve])
    result['mi'] = mi
    return result    

def demonstrate_separation_advantage(num_meanings=100,
                                     num_meanings_per_word=2,
                                     num_signals=10,
                                     num_signals_per_morpheme=4,
                                     source='zipf',
                                     with_delimiter='both',
                                     **kwds):
    if source == 'zipf':
        flat_source = s.zipf_mandelbrot(num_meanings**num_meanings_per_word, **kwds)
    elif source == 'rem':
        flat_source = s.rem(num_meanings**num_meanings_per_word, **kwds)
    source = s.factor(flat_source, num_meanings, num_meanings_per_word)
    code = c.random_code(num_meanings, num_signals, num_signals_per_morpheme)
    signal = c.word_probabilities(source, code, with_delimiter=with_delimiter, marginalize=False)
    meanings = itertools.product(*[range(num_meanings)]*num_meanings_per_word)
    signal['m'] = list(meanings)
    for i in range(num_meanings_per_word):
        signal[i] = signal['m'].map(lambda x: x[i])
    
    signal['form_sys_scrambled'] = signal['form'].map(il.scramble_form)
    signal['form_sys_dscrambled'] = signal['form'].map(il.DeterministicScramble().shuffle)
    signal['form_sys_eo'] = signal['form'].map(il.even_odd)
    signal['form_sys_oi'] = signal['form'].map(il.outward_in)
    signal['form_quasisys'] = np.random.permutation(signal['form'].values)
    signal.rename({'form': 'form_sys'}, axis=1, inplace=True)

    flat_source = source.flatten()
    syncretic_code = c.random_code(len(flat_source), num_signals, num_signals_per_morpheme * num_meanings_per_word)
    syncretic_signal = c.word_probabilities(flat_source, syncretic_code, with_delimiter=with_delimiter, marginalize=False)

    signal = pd.merge(signal, syncretic_signal.rename({'form': 'form_syn'}, axis=1))

    for i in range(num_meanings_per_word):
        marginal = signal[[i, 'probability']].groupby(i).sum().reset_index().rename({'probability': 'p'+str(i)}, axis=1)
        signal = pd.merge(signal, marginal)

    return signal

def demonstrate_separation_exceptions(k=10, **kwds):
    # These don't work!?!
    df = demonstrate_separation_advantage(**kwds)
    n = len(df)
    random_indices = random.sample(range(n), k)

    # Penalty of replacing the top k most frequent forms with their syncretic equivalents
    # vs. the bottom k, and random k
    dfp = df.sort_values('probability')
    dfp['freqsyn'] = np.hstack([dfp['form_sys'][:(n-k)], dfp['form_syn'][-k:]])
    dfp['infreqsyn'] = np.hstack([dfp['form_syn'][:k], dfp['form_sys'][-(n-k):]])
    dfp['rfreqsyn'] = dfp['form_sys'].copy()
    dfp['rfreqsyn'][random_indices] = dfp['form_syn'][random_indices]

    print(
        il.score(il.ee, dfp['form_sys'], dfp['probability']),
        il.score(il.ee, dfp['form_syn'], dfp['probability']),
        il.score(il.ee, dfp['freqsyn'], dfp['probability']),
        il.score(il.ee, dfp['infreqsyn'], dfp['probability']),
        il.score(il.ee, dfp['rfreqsyn'], dfp['probability']),
    )

    df['pmi'] = np.log(df['probability']) - np.log(df['p0']) - np.log(df['p1'])
    dfm = df.sort_values('pmi')
    dfm['high_mi_syn'] = np.hstack([dfm['form_sys'][:(n-k)], dfm['form_syn'][-k:]])
    dfm['low_mi_syn'] = np.hstack([dfm['form_syn'][:k], dfm['form_sys'][-(n-k):]])
    dfm['r_mi_syn'] = dfm['form_sys'].copy()
    dfm['r_mi_syn'][random_indices] = dfm['form_syn'][random_indices]

    print(
        il.score(il.ee, dfm['form_sys'], dfm['probability']),
        il.score(il.ee, dfm['form_syn'], dfm['probability']),
        il.score(il.ee, dfm['high_mi_syn'], dfm['probability']),
        il.score(il.ee, dfm['low_mi_syn'], dfm['probability']),
        il.score(il.ee, dfm['r_mi_syn'], dfm['probability']),
    )

def demonstrate_contiguity_preference(num_meanings=10,
                                      num_meanings_per_word=2,
                                      num_signals=2,
                                      num_signals_per_morpheme=4,
                                      with_delimiter='both',
                                      source='zipf',
                                      **kwds):
    """ Contiguous codes are preferred over interleaving codes.
    
    For a language where meaning factors into num_meanings_per_word components,
    each of which is mapped to a morpheme of length num_signals_per_morpheme,
    compare the MS tradeoff for variants applying a deterministic shuffle to strings.

    """
    # Source is zipfian and split into k components each with M possible values
    if source == 'zipf':
        flat_source = s.zipf_mandelbrot(num_meanings**num_meanings_per_words, **kwds)
    elif source == 'rem':
        flat_source = s.rem(num_meanings**num_meanings_per_word, **kwds)
    source = s.factor(flat_source, num_meanings, num_meanings_per_word)

    # Code is random, not necessarily one-to-one.
    code = c.random_code(num_meanings, num_signals, num_signals_per_morpheme)

    # Get signal probabilities.
    signal = c.word_probabilities(source, code, with_delimiter=with_delimiter)
    
    # Go through global permutations  
    def gen():
        for perm in itertools.permutations(range(num_meanings_per_word*num_signals_per_morpheme)):
            reordered = signal['form'].map(lambda s: il.reorder_form(s, perm))
            counts = il.counts_from_sequences(reordered, weights=signal['probability']) # need to exp?
            curves = il.mle_curves_from_counts(counts['count'], counts['x_{<t}'])
            ms = il.ms_auc(curves)
            ee = il.ee(curves)
            contiguous = c.is_contiguous(num_meanings_per_word, num_signals_per_morpheme, perm)
            yield {
                'is_contiguous': contiguous,
                'ms': ms,
                'ee': ee,
                'perm': perm,
            }
    df = pd.DataFrame(gen())
    fig, axs = plt.subplots(2)
    fig.suptitle("Advantage of contiguous orders")
    dfe = df.sort_values(['ee']).reset_index()
    dfm = df.sort_values(['ms']).reset_index()    
    axs[0].scatter(dfe.index, dfe['ee'], c=dfe['is_contiguous'])
    axs[0].set(ylabel="Excess entropy")
    axs[1].scatter(dfm.index, dfm['ms'], c=dfm['is_contiguous'])
    axs[1].set(ylabel="MS Tradeoff", xlabel="Permutation order")

    return source, signal, df

def num_discontinuities(perm):
    set1 = {0, 1, 2, 3}
    set2 = {4, 5, 6, 7}
    d = 0
    which = perm[0] in set1
    maybe = False
    for i in perm:
        if (i in set1) != which:  # still in same morpheme
            d += 1
            which = i in set1
    return d

def summarize_mle(source, code, with_delimiter='both'):
    signal = c.form_probabilities_np(source, code, with_delimiter=with_delimiter)
    curves = il.curves_from_sequences(signal['form'], signal['probability'])
    return curves

def strong_combinatoriality(num_morphemes=4,
                            num_parts=4,
                            morpheme_length=4,
                            vocab_size=2,
                            with_delimiter='left',
                            perturbation_size=0,
                            source='zipf',
                            **kwds):
    # Independent perturbed-iid morphemes. Is strong or weak systematicity better?
    # Intuition: Strong systematicity means your short-range entropy doesn't depend on which word you're in, so it's better.
    
    # With left-delimiter only, strong is better. With both delimiters, weak is better:
    # Strong systematicity is better at low t. 
    # Weak systematicity is better at high t.
    # Crossover around t=morpheme_length. Both reach h_t=h at t=3*morpheme_length. 

    # Shouldn't strong systematicity yield a lower entropy rate at low t? it does...
    # Why does strong systematicity give a higher entropy rate at high t?
    # -> Because it does not give strong cues to where you are in the signal!
    
    num_meanings = num_morphemes**num_parts

    if source == 'zipf':
        morpheme_source = s.zipf_mandelbrot(num_morphemes, **kwds)
        np.random.shuffle(morpheme_source)
    elif source == 'rem':
        morpheme_source = s.rem(num_morphemes, **kwds)
        
    source = np.array([0])
    for k in range(num_parts):
        source = s.product_distro(
            source,
            scipy.special.softmax(
                np.log(morpheme_source) + perturbation_size*np.random.randn(num_morphemes)
            )
        )
    source = source.reshape((num_morphemes,)*num_parts)

    # Build codes
    strong_code = c.random_code(num_morphemes, vocab_size, morpheme_length)
    holistic_code = c.random_code(num_meanings, vocab_size, morpheme_length*num_parts)

    # Weak systematic code is permutations of the strong code; this controls entropy rate
    shuffles = random.sample(list(itertools.permutations(range(num_morphemes))), num_parts)
    weak_codes = [strong_code[list(shuffle)] for shuffle in shuffles]

    # Free order lowers the entropy rate, why? It creates collisions. Avoid using positional coding?

    signals = dict([
        # Strongly systematic code:
        ('strong', c.word_probabilities(source, strong_code, with_delimiter=with_delimiter)),

        # Strongly systematic code with positional marking
        ('strong_positional', c.word_probabilities(source, strong_code, encode=c.encode_contiguous_positional, with_delimiter=with_delimiter)),

        ('free_order', c.word_probabilities(source, strong_code, encode=c.encode_contiguous_random_order, with_delimiter=with_delimiter)),
        ('free_positional', c.word_probabilities(source, strong_code, encode=c.encode_contiguous_positional_random_order, with_delimiter=with_delimiter)),
        
        ('weak', c.word_probabilities(source, weak_codes, encode=c.encode_weak_contiguous, with_delimiter=with_delimiter)),
        ('holistic', c.word_probabilities(source.flatten(), holistic_code, with_delimiter=with_delimiter)),
    ])

    def gen():
        for name, signal in signals.items():
            counts = il.counts_from_sequences(signal['form'], weights=signal['probability'])
            curves = il.mle_curves_from_counts(counts['count'], counts['x_{<t}'])
            curves['type'] = name
            yield curves
            
    return pd.concat(list(gen()))

def combinatoriality(morpheme_length=4, vocab_size=2, with_delimiter='both', source='zipf', **kwds):
    if source == 'zipf':
        source = s.zipf_mandelbrot(256, **kwds)
        np.random.shuffle(source)
    elif source == 'rem':
        source = s.rem(256, **kwds)
        
    # 256 = 4 x 4 x 4
    codes = [
        (2, 8, c.random_code(2, vocab_size, morpheme_length*1)),     # 2^8
        (4, 4, c.random_code(4, vocab_size, morpheme_length*2)),     # 4^4
        (16, 2, c.random_code(16, vocab_size, morpheme_length*4)),   # 16**2
        (256, 1, c.random_code(256, vocab_size, morpheme_length*8)), # 256**1
    ]
    def gen():
        for num_morphemes, num_parts, code in codes:
            shape = (num_morphemes,)*num_parts
            signal = c.word_probabilities(
                source.reshape(shape),
                code,
                with_delimiter=with_delimiter
            )
            counts = il.counts_from_sequences(signal['form'], weights=signal['probability'])
            curves = il.mle_curves_from_counts(counts['count'], counts['x_{<t}'])
            curves['num_morphemes'] = num_morphemes
            curves['num_parts'] = num_parts
            yield curves
    return pd.concat(list(gen()))
        
def id_vs_cnot(mi=1/2):
    source = s.mi_mix(mi)
    id = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    cnot = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ])
    df1 = summarize_mle(source, id)
    df1['type'] = 'id'
    df2 = summarize_mle(source, cnot)
    df2['type'] = 'cnot'
    return pd.concat([df1, df2])

def id_vs_cnot_sources(num_samples=1000, source='rem', **kwds):
    id = np.array([
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    ])
    cnot12 = np.array([
        [0, 2],
        [0, 3],
        [1, 3],
        [1, 2],
    ])

    cnot21 = np.array([
        [0, 2],
        [1, 3],
        [1, 2],
        [0, 3],
    ])

    def gen():
        for i in tqdm.tqdm(range(num_samples)):
            if source == 'zipf':
                source_flat = s.zipf_mandelbrot(4, **kwds)
            elif source == 'rem':
                source_flat = s.rem(4, **kwds)
            source_factored = source_flat.reshape(2,2)
            source_x = source_factored.sum(-1)
            source_y = source_factored.sum(-2)
            yield {
                'mi': s.mi(source_factored),
                'id': summarize_mle(source_flat, id, **kwds)['H_M_lower_bound'].max(),
                'cnot12': summarize_mle(source_flat, cnot12, **kwds)['H_M_lower_bound'].max(),
                'cnot21': summarize_mle(source_flat, cnot21, **kwds)['H_M_lower_bound'].max(),
                'h1': s.entropy(source_x),
                'h2': s.entropy(source_y),
            }
    return pd.DataFrame(gen())

def id_vs_cnot4(i23=.9, i14=0, **kwds):
    source = s.product_distro(s.flip(1/2), s.product_distro(s.mi_mix(i23), s.flip(1/2)))

    def langs():
        yield 'syst1234', np.array([  # L1 L2 L3 L4, local and systematic
            [0, 2, 4, 7],
            [0, 2, 4, 8],
            [0, 2, 5, 7],
            [0, 2, 5, 8],
            [0, 3, 4, 7],
            [0, 3, 4, 8],
            [0, 3, 5, 7],
            [0, 3, 5, 8],
            [1, 2, 4, 7],
            [1, 2, 4, 8],
            [1, 2, 5, 7],
            [1, 2, 5, 8],
            [1, 3, 4, 7],
            [1, 3, 4, 8],
            [1, 3, 5, 7],
            [1, 3, 5, 8],        
        ])

        yield 'shuffle', np.array([
            [1, 2, 5, 7],
            [1, 2, 4, 8],
            [0, 3, 5, 7],
            [1, 3, 5, 8],
            [0, 3, 5, 8],
            [1, 2, 5, 8],
            [1, 3, 4, 8],
            [1, 3, 5, 7],
            [0, 2, 5, 8],
            [1, 3, 4, 7],
            [0, 2, 4, 7],
            [0, 2, 4, 8],
            [0, 3, 4, 8],
            [0, 2, 5, 7],
            [0, 3, 4, 7],
            [1, 2, 4, 7]
        ])

        yield 'syst_free', np.array([
            [2, 7, 0, 4],
            [2, 8, 0, 4],
            [2, 7, 0, 5],
            [5, 8, 0, 2],
            [7, 3, 0, 4],
            [0, 4, 3, 8],
            [3, 7, 0, 5],
            [0, 3, 8, 5],
            [1, 4, 2, 7],
            [1, 8, 2, 4],
            [2, 7, 5, 1],
            [8, 5, 2, 1],
            [1, 7, 4, 3],
            [8, 4, 3, 1],
            [5, 1, 3, 7],
            [5, 3, 8, 1]
        ])
        
        yield 'syst2143', np.array([  # L2 L1 L4 L3, nonlocal and systematic
            [2, 0, 7, 4],
            [2, 0, 8, 4],
            [2, 0, 7, 5],
            [2, 0, 8, 5],
            [3, 0, 7, 4],
            [3, 0, 8, 4],
            [3, 0, 7, 5],
            [3, 0, 8, 5],
            [2, 1, 7, 4],
            [2, 1, 8, 4],
            [2, 1, 7, 5],
            [2, 1, 8, 5],
            [3, 1, 7, 4],
            [3, 1, 8, 4],
            [3, 1, 7, 5],
            [3, 1, 8, 5],        
        ])

        yield 'syst2134', np.array([  # L2 L1 L3 L4, interleaved and systematic
            [2, 0, 4, 7],
            [2, 0, 4, 8],
            [2, 0, 5, 7],
            [2, 0, 5, 8],
            [3, 0, 4, 7],
            [3, 0, 4, 8],
            [3, 0, 5, 7],
            [3, 0, 5, 8],
            [2, 1, 4, 7],
            [2, 1, 4, 8],
            [2, 1, 5, 7],
            [2, 1, 5, 8],
            [3, 1, 4, 7],
            [3, 1, 4, 8],
            [3, 1, 5, 7],
            [3, 1, 5, 8],        
        ])    

        yield 'fuse23', np.array([   # L1 L23 L4 -- good.
            [0, 2, 4, 7],
            [0, 2, 4, 8],
            [0, 2, 5, 7],
            [0, 2, 5, 8],
            [0, 3, 5, 7],
            [0, 3, 5, 8],
            [0, 3, 4, 7],
            [0, 3, 4, 8],
            [1, 2, 4, 7],
            [1, 2, 4, 8],
            [1, 2, 5, 7],
            [1, 2, 5, 8],
            [1, 3, 5, 7],
            [1, 3, 5, 8],
            [1, 3, 4, 7],
            [1, 3, 4, 8],
        ])

        yield 'fuse23_free', np.array([   # L1 L23 L4 -- good.
            [0, 2, 4, 7],
            [0, 2, 8, 4],
            [0, 7, 5, 2],
            [0, 2, 8, 5],
            [3, 5, 0, 7],
            [8, 0, 3, 5],
            [0, 4, 7, 3],
            [0, 3, 8, 4],
            [2, 4, 1, 7],
            [2, 8, 1, 4],
            [7, 1, 2, 5],
            [2, 5, 8, 1],
            [1, 3, 5, 7],
            [8, 1, 5, 3],
            [3, 1, 4, 7],
            [3, 8, 4, 1],
        ])

        yield 'fuse12', np.array([  # L12 L3 L4 -- noise controls a meaningful bit, bad
            [0, 2, 4, 7],
            [0, 2, 4, 8],
            [0, 2, 5, 7],
            [0, 2, 5, 8],
            [0, 3, 4, 7],
            [0, 3, 4, 8],
            [0, 3, 5, 7],
            [0, 3, 5, 8],
            [1, 3, 4, 7],
            [1, 3, 4, 8],
            [1, 3, 5, 7],
            [1, 3, 5, 8],
            [1, 2, 4, 7],
            [1, 2, 4, 8],
            [1, 2, 5, 7],
            [1, 2, 5, 8],                
        ])

        yield 'fuse34', np.array([  # CNOT(3,4) --- a meaningful bit controls noise, bad but not as bad as CNOT(1,2).
            [0, 2, 4, 7],
            [0, 2, 4, 8],
            [0, 2, 5, 8],
            [0, 2, 5, 7],
            [0, 3, 4, 7],
            [0, 3, 4, 8],
            [0, 3, 5, 8],
            [0, 3, 5, 7],
            [1, 2, 4, 7],
            [1, 2, 4, 8],
            [1, 2, 5, 8],
            [1, 2, 5, 7],
            [1, 3, 4, 7],
            [1, 3, 4, 8],
            [1, 3, 5, 8],
            [1, 3, 5, 7],        
    ])

        yield 'fuse13', np.array([  # (L13)_1 L2 (L13)_2 L4 -- a noise bit controls a nonlocal meaningful bit. should be very bad.
            [0, 2, 4, 7],
            [0, 2, 4, 8],
            [0, 2, 5, 7],
            [0, 2, 5, 8],
            [0, 3, 4, 7],
            [0, 3, 4, 8],
            [0, 3, 5, 7],
            [0, 3, 5, 8],
            [1, 2, 5, 7],
            [1, 2, 5, 8],
            [1, 2, 4, 7],
            [1, 2, 4, 8],
            [1, 3, 5, 7],
            [1, 3, 5, 8],
            [1, 3, 4, 7],
            [1, 3, 4, 8],                    
        ])

        yield 'fuse23_interleaved', np.array([  # (L23)_1 L_1 (L_23)_2 L_4
            [2, 0, 4, 7],
            [2, 0, 4, 8],
            [2, 0, 5, 7],
            [2, 0, 5, 8],
            [3, 0, 5, 7],
            [3, 0, 5, 8],
            [3, 0, 4, 7],
            [3, 0, 4, 8],
            [2, 1, 4, 7],
            [2, 1, 4, 8],
            [2, 1, 5, 7],
            [2, 1, 5, 8],
            [3, 1, 5, 7],
            [3, 1, 5, 8],
            [3, 1, 4, 7],
            [3, 1, 4, 8],        
        ])

        yield 'fuse23_nonlocal', np.array([  # (L23)_1 L_1 L_4 (L_23)_2
            [2, 0, 7, 4],
            [2, 0, 8, 4],
            [2, 0, 7, 5],
            [2, 0, 8, 5],
            [3, 0, 7, 5],
            [3, 0, 8, 5],
            [3, 0, 7, 4],
            [3, 0, 8, 4],
            [2, 1, 7, 4],
            [2, 1, 8, 4],
            [2, 1, 7, 5],
            [2, 1, 8, 5],
            [3, 1, 7, 5],
            [3, 1, 8, 5],
            [3, 1, 7, 4],
            [3, 1, 8, 4],        
        ])

    def process(nl):
        name, lang = nl
        df = summarize_mle(source, lang, **kwds)
        df['type'] = name
        return df

    the_langs = dict(langs())
    df = pd.concat(map(process, the_langs.items()))
    # Result for EE:
    # For systematic langs: local < interleaved < nonlocal, good
    # For nonsystematic langs: fuse23 < fuse34 < fuse21 = fuse31.


    # Intuition for why fuse23_nonlocal is not worse than fuse23:
    # In these languages, "E" means "the more common M3 given M2" and "F" means "the less common M3 given M2"
    # Thus, entropy of EF is *NOT* reduced by knowing CD! There is actually *LESS* MI here. (decorrelation)
    # Therefore, since MI is zero, nonlocality isn't a problem.
    # So why does context reduce MI at all? Because 1 character of context suffices to establish position in the sequence.
    
    return df, source, the_langs

def three_sweep(i12=0, i23=0.9, **kwds):
    source = s.mi_mix3(i12, i23)
    id_code = np.array([
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 4],
        [0, 3, 5],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 4],
        [1, 3, 5],
    ])
    the_range = range(8)
    def gen():
        for permutation in tqdm.tqdm(list(itertools.permutations(the_range))):
            code = id_code[list(permutation)]
            curves = summarize_mle(source, code, **kwds)
            yield permutation, curves['H_M_lower_bound'].max(), code
    permutations, ees, codes = zip(*gen())
    return pd.DataFrame({'permutation': permutations, 'ee': ees, 'code': codes})

def id_vs_cnot3(i12=0, i23=0.9, **kwds):
    # I_12 = 0.
    # I_23 = 0.5
    source = s.mi_mix3(i12, i23)
    
    id_code = np.array([
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 4],
        [0, 3, 5],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 4],
        [1, 3, 5],
    ])

    fuse12 = np.array([  # CNOT(1,2)
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 4],
        [0, 3, 5],
        [1, 3, 4],
        [1, 3, 5],
        [1, 2, 4],
        [1, 2, 5],        
    ])

    fuse23 = np.array([  # CNOT(2,3)
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 5],
        [0, 3, 4],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 5],
        [1, 3, 4],        
    ])

    fuse23_nonlocal = np.array([
        [2, 0, 4],
        [2, 0, 5],
        [3, 0, 5],
        [3, 0, 4],
        [2, 1, 4],
        [2, 1, 5],
        [3, 1, 5],
        [3, 1, 4],                
    ])

    df1 = summarize_mle(source, id_code, **kwds)
    df1['type'] = 'id'
    df2 = summarize_mle(source, fuse12, **kwds)
    df2['type'] = 'cnot12'
    df3 = summarize_mle(source, fuse23, **kwds)
    df3['type'] = 'cnot23'

    df = pd.concat([df1, df2, df3])

    plot = (
        ggplot(df, aes(x='t+1', y='h_t/np.log(2)', color='type'))
        + geom_line(size=1.1)
        + labs(x='Markov order t', y='Markov entropy rate hₜ', color='')
        + xlim(None, 3)
        + theme_classic()
        + guides(color=False)
        
        + geom_text(aes(x=2.35, y=2.4, label='"L₁·L₂₃"'), color='black')
        + geom_text(aes(x=2.35, y=2.3, label='"E=1.83 bits"'), color='black')
        + geom_segment(aes(x=2.1, y=2.37, xend=1.17, yend=2.1), color='black')

        + geom_text(aes(x=2.35, y=1.9, label='"L₁·L₂·L₃"'), color='black')
        + geom_text(aes(x=2.35, y=1.8, label='"E=1.58 bits"'), color='black')
        + geom_segment(aes(x=2.1, y=1.86, xend=1.62, yend=1.5), color='black')

        + geom_text(aes(x=2.35, y=1.4, label='"L₁₂·L₃"'), color='black')
        + geom_text(aes(x=2.35, y=1.3, label='"E=2.06 bits"'), color='black')
        + geom_segment(aes(x=2.1, y=1.35, xend=1.92, yend=1.16), color='black')                
    )    
    
    return df, plot



def huffman_systematic_vs_not(first_prob=1/3, **kwds):
    # Meaning has two components, M_1 x M_2, with zero MI
    # It's just a product of an independent (1/3,2/3) * (1/2,1/4,1/8,1/8)
    # There are multiple legitimate Huffman codes for this source, one systematic and one not.
    # The systematic Huffman code has a better MS tradeoff due to less long-term dependencies -- conditional on delimiters. Why?
    # Should use with_delimiter='left'
    source = s.product_distro(s.flip(first_prob), np.array([1/2, 1/4, 1/8, 1/8]))
    systematic_huffman = np.array([
        [0, 0],
        [0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0],
        [1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ])
    nonsystematic_huffman = np.array([
        [0, 0],
        [1, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [1, 0],
        [1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
    ])
    df1 = summarize_mle(source, systematic_huffman, **kwds)
    df1['type'] = 'systematic'
    df2 = summarize_mle(source, nonsystematic_huffman, **kwds)
    df2['type'] = 'nonsystematic'
    df = pd.concat([df1, df2])

    plot = (
        ggplot(df, aes(x='t+1', y='h_t/np.log(2)', color='type'))
        + geom_line(size=1.1) 
        + theme_minimal() 
        + guides(color=False) 
        + labs(x='Markov order t', y='Markov entropy rate hₜ (bits)')
        + theme(legend_position="top")
        + xlim(None,4) 
        + geom_text(aes(x=3.5, y=0.988, label='"nonsystematic"'), color='black')
        + geom_text(aes(x=3.5, y=0.987, label='"E=0.039 bits"'), color='black')
        + geom_text(aes(x=3.5, y=0.982, label='"systematic"'), color='black')
        + geom_text(aes(x=3.5, y=0.981, label='"E=0.019 bits"'), color='black')
        + geom_segment(aes(x=3.1, xend=2.41, y=0.9875, yend=0.983), color='black')
        + geom_segment(aes(x=3.1, xend=1.76, y=0.9815, yend=0.9775), color='black') 
    )
    return df, plot

def huffman_vs_separable(mi=1):
    # Given meanings that can be decomposed into two parts, M_1 x M_2,
    # withdf H[M_1] = 2, H[M_2] = 2, and I[M_1 : M_2] = 1,
    # when is it better to use a 3-bit code vs. a 4-bit separable code?
    source = np.ones(8)/8
    compressed_code = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    separable_local_code = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],        
    ])
    separable_nonlocal_code = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
    ])
    df1 = summarize_mle(source, compressed_code)
    df1['type'] = 'compressed'
    df2 = summarize_mle(source, separable_local_code)
    df2['type'] = 'separable_local'
    df3 = summarize_mle(source, separable_nonlocal_code)
    df3['type'] = 'separable_nonlocal'
    return pd.concat([df1, df2, df3])

def shuffled(xs):
    xs = list(xs)
    random.shuffle(xs)
    return xs

def locality(V=5, S=2, l=5, with_delimiter='both', **kwds):
    # dhd is consistently best with S=20, l=5: unambiguous word boundaries.
    # with S=2, l=5 and left delimiter, dhd or hdd is the best, inconsistently.
    # -> The hdd advantage has something to do with phonotactics...
    source, meanings, mis = s.dependents(V=V, k=2, **kwds)
    codes = {
        'hdd': [
            (meaning['h'],) + tuple(meaning.values())[1:]
            for meaning in meanings
        ],
        'hdd_bad': [
            (meaning['h'],) + tuple(reversed(list(meaning.values())[1:]))
            for meaning in meanings
        ],
        'dhd': [
            (meaning['d0'], meaning['h'], meaning['d1'])
            for meaning in meanings
        ],
        'hdd_free': [
            (meaning['h'],) + tuple(shuffled(list(meaning.values())[1:]))
            for meaning in meanings
        ]
    }
    systematic, codebook = c.random_systematic_code([m.values() for m in meanings], S, l, unique=True)
    def gen():
        for name, order in codes.items():
            forms = c.form_probabilities(source, order, systematic, with_delimiter=with_delimiter)
            curves = il.curves_from_sequences(forms['form'], forms['p'])
            curves['type'] = name
            yield curves

    return pd.concat(list(gen())), mis, codebook

def mansfield_kemp(S=50, l=1, with_delimiter='both'):
    source, meanings = s.mansfield_kemp_source()
    systematic, codebook = c.random_systematic_code(meanings, S, l, unique=True)
    parts = "nAND"

        
    
    def codes():
        for order in itertools.permutations(range(4)):
            label = "".join(parts[i] for i in order)
            grammar = [
                tuple(meaning[i] for i in order)
                for meaning in meanings
            ]
            yield label, grammar
    the_codes = dict(codes())
    def gen():
        for name, order in tqdm.tqdm(the_codes.items()):
            forms = c.form_probabilities(source, order, systematic, with_delimiter=with_delimiter)
            curves = il.curves_from_sequences(forms['form'], forms['p'])
            curves['type'] = name
            yield curves


    # predicted typology
    typology = pd.DataFrame({
        'type': "nAND DNAn DnAN DNnA NnAD nADN nDAN nNAD DnNA DAnN nDNA NAnD AnND NnDA NDAn AnDN DANn nNDA NADn NDnA ADnN ADNn ANDn ANnD".split(),
        'af': [43.50, 36.62, 28.34, 21.18, 15.33, 14.78, 9.00, 9.00, 8.77, 6.11, 4.67, 4.00, 3.00, 3.00, 3.00, 2.49, 2.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        'num_genera': [84, 57, 38, 31, 28, 19, 11, 9, 10, 8, 5, 5, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0],
    })    
    groups = [
        frozenset("DAnN NnAD".split()),
        frozenset("DNnA AnND".split()),
        frozenset("DNAn nAND".split()),
        frozenset("DnAN NAnD".split()),
        frozenset("NDnA AnDN".split()),
        frozenset("NDAn nADN".split()),
        frozenset("DANn nNAD".split()),
        frozenset("DnNA ANnD".split()),
        frozenset("NnDA ADnN".split()),
        frozenset("nDAN NADn".split()),
        frozenset("nNDA ADNn".split()),
        frozenset("nDNA ANDn".split()),
    ]
    typology['group'] = typology['type'].map(lambda t: rfutils.first(s for s in groups if t in s))
    af = typology[['group', 'af', 'num_genera']].drop_duplicates().groupby('group').sum().reset_index()
    af.columns = ['group', 'af_sum', 'num_genera_sum']
    return pd.concat(list(gen())).merge(typology.merge(af)), codebook
            
def aanaa(S=50, l=1, with_delimiter='both'):
    # For A*N sequences, 
    # 1. Class-level information locality: higher mi adjective classes go closer. -- Yes, if MI differential is big enough.
    # 2. Consistent AAN better than Gildea & Temperley inward-outward order. -- Yes, and higher ER, if differential is big enough.
    # 3. Class consistency is more important than individual pmi with noun. - Yes.
    # 4. Class-consistent ANA is better than G&T-style ANA. -- doesn't seem to work.

    # with 2-3 classes, each of which has some MI with the noun.
    # show that the code which orders by MI is best.
    # then show consistency is good -- an inconsistent inward-outward ordering is worse.


    # classes are defined by I[A:B|N] = 0. Conditionally independent given the N.
    
    source, meanings, pmis, mis = s.astarn(5, 5, 3)

    # two representations of meaning wrt order of features
    highest_first = [
        (meaning['n'],) + tuple(sorted(list(meaning.values())[1:]))
        for meaning in meanings
    ]
    lowest_first = [
        (meaning['n'],) + tuple(reversed(sorted(list(meaning.values())[1:])))
        for meaning in meanings
    ]
    gt_optimal = [
        (meaning[3], meaning[1], meaning[0], meaning[2]) if len(meaning) == 4 else
        (meaning[1], meaning[0], meaning[2]) if len(meaning) == 3 else
        (meaning[0], meaning[1]) if len(meaning) == 2 else (meaning[0],)
        for meaning in highest_first
    ]
    gt_pessimal = [
        (meaning[3], meaning[1], meaning[0], meaning[2]) if len(meaning) == 4 else
        (meaning[1], meaning[0], meaning[2]) if len(meaning) == 3 else
        (meaning[1], meaning[0]) if len(meaning) == 2 else (meaning[0],)
        for meaning in lowest_first
    ]
    by_pmi_naa = [
        (meaning['n'],) + tuple(a for pmi, a in reversed(sorted(zip(pmi.values(), list(meaning.values())[1:]))))
        for pmi, meaning in zip(pmis, meanings)
    ]
    anti_by_pmi_naa = [
        (meaning['n'],) + tuple(a for pmi, a in sorted(zip(pmi.values(), list(meaning.values())[1:])))
        for pmi, meaning in zip(pmis, meanings)
    ]
    def consistent_ana(meaning): # Put class 0 (highest MI) always before noun
        adjectives = sorted(list(meaning.values())[1:])
        if adjectives and adjectives[0][0] == 0:
            yield adjectives[0]
            yield meaning['n']
            yield from adjectives[1:]
        else:
            yield meaning['n']
            yield from adjectives
    consistent_ana = [tuple(consistent_ana(meaning)) for meaning in meanings]

    # make a systematic unambiguous code
    systematic, codebook = c.random_systematic_code(highest_first, S, l, unique=True)

    # These represent different orderings of the same code
    forms_local = c.form_probabilities(source, highest_first, systematic, with_delimiter=with_delimiter)
    forms_nonlocal = c.form_probabilities(source, lowest_first, systematic, with_delimiter=with_delimiter)
    forms_gt = c.form_probabilities(source, gt_optimal, systematic, with_delimiter=with_delimiter)
    forms_antigt = c.form_probabilities(source, gt_pessimal, systematic, with_delimiter=with_delimiter)
    forms_pmi = c.form_probabilities(source, by_pmi_naa, systematic, with_delimiter=with_delimiter)
    forms_antipmi = c.form_probabilities(source, anti_by_pmi_naa, systematic, with_delimiter=with_delimiter)
    forms_consistent = c.form_probabilities(source, consistent_ana, systematic, with_delimiter=with_delimiter)
    

    curves = il.curves_from_sequences(forms_local['form'], forms_local['p'])
    curves['type'] = 'local'
    curves2 = il.curves_from_sequences(forms_nonlocal['form'], forms_nonlocal['p'])
    curves2['type'] = 'nonlocal'
    curves3 = il.curves_from_sequences(forms_gt['form'], forms_gt['p'])
    curves3['type'] = 'gt'
    curves4 = il.curves_from_sequences(forms_antigt['form'], forms_antigt['p'])
    curves4['type'] = 'antigt'
    curves5 = il.curves_from_sequences(forms_pmi['form'], forms_pmi['p'])
    curves5['type'] = 'pmi'
    curves6 = il.curves_from_sequences(forms_antipmi['form'], forms_antipmi['p'])
    curves6['type'] = 'antipmi'
    curves7 = il.curves_from_sequences(forms_consistent['form'], forms_consistent['p'])
    curves7['type'] = 'consistent_ana'

    return pd.concat([curves, curves2, curves3, curves4, curves5, curves6, curves7]), mis

    

    

    
    




