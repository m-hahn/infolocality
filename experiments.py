import sys
import math
import itertools
import functools
import random
from collections import Counter

import tqdm
import numpy as np
import pandas as pd
import scipy.special
import scipy.optimize
import matplotlib.pyplot as plt
from plotnine import *

import infoloc as il
import sources as s
import shuffles as sh
import codes as c
import featurecorr as f

def fusion_advantage(mi=1/2):
    p = s.mi_mix(mi)
    mi = s.mi(p.reshape(2,2))
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
    mi = s.mi(p.reshape(2*2,2))
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
    meanings = c.cartesian_indices(num_meanings, num_meanings_per_word)
    signal['m'] = list(meanings)
    for i in range(num_meanings_per_word):
        signal[i] = signal['m'].map(lambda x: x[i])
    
    signal['form_sys_scrambled'] = signal['form'].map(sh.scramble_form)
    signal['form_sys_dscrambled'] = signal['form'].map(sh.DeterministicScramble().shuffle)
    signal['form_sys_eo'] = signal['form'].map(sh.even_odd)
    signal['form_sys_oi'] = signal['form'].map(sh.outward_in)
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

def demonstrate_separation_exceptions(k=10, metric=il.ee, **kwds):
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
        il.score(metric, dfp['form_sys'], dfp['probability']),
        il.score(metric, dfp['form_syn'], dfp['probability']),
        il.score(metric, dfp['freqsyn'], dfp['probability']),
        il.score(metric, dfp['infreqsyn'], dfp['probability']),
        il.score(metric, dfp['rfreqsyn'], dfp['probability']),
    )

    df['pmi'] = np.log(df['probability']) - np.log(df['p0']) - np.log(df['p1'])
    dfm = df.sort_values('pmi')
    dfm['high_mi_syn'] = np.hstack([dfm['form_sys'][:(n-k)], dfm['form_syn'][-k:]])
    dfm['low_mi_syn'] = np.hstack([dfm['form_syn'][:k], dfm['form_sys'][-(n-k):]])
    dfm['r_mi_syn'] = dfm['form_sys'].copy()
    dfm['r_mi_syn'][random_indices] = dfm['form_syn'][random_indices]

    print(
        il.score(metric, dfm['form_sys'], dfm['probability']),
        il.score(metric, dfm['form_syn'], dfm['probability']),
        il.score(metric, dfm['high_mi_syn'], dfm['probability']),
        il.score(metric, dfm['low_mi_syn'], dfm['probability']),
        il.score(metric, dfm['r_mi_syn'], dfm['probability']),
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
        flat_source = s.zipf_mandelbrot(num_meanings**num_meanings_per_word, **kwds)
    elif source == 'rem':
        flat_source = s.rem(num_meanings**num_meanings_per_word, **kwds)
    source = s.factor(flat_source, num_meanings, num_meanings_per_word)

    # Code is random, not necessarily one-to-one.
    code = c.random_code(num_meanings, num_signals, num_signals_per_morpheme)

    # Get signal probabilities.
    signal = c.word_probabilities(source, code, with_delimiter=with_delimiter)
    
    # Go through global permutations
    print(math.factorial(num_meanings_per_word*num_signals_per_morpheme), file=sys.stderr)
    def gen():
        for perm in tqdm.tqdm(itertools.permutations(range(num_meanings_per_word*num_signals_per_morpheme))):
            reordered = signal['form'].map(lambda s: sh.reorder_form(s, perm))
            curves = il.curves_from_sequences(reordered, weights=signal['probability']) # need to exp?
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

def summarize(source, code, with_delimiter='both'):
    signal = c.form_probabilities_np(source, code, with_delimiter=with_delimiter)
    curves = il.curves_from_sequences(signal['form'], signal['probability'])
    return curves

# when is systematicity *bad*?
# -- In ABC, if BC have MI, A.BC is better than A.B.C

def strong_combinatoriality_sweep(min_coupling=0, max_coupling=10, num_steps=10, num_samples=10, **kwds):
    perturbations = np.linspace(min_coupling, max_coupling, num_steps)
    def gen():
        for perturbation in tqdm.tqdm(perturbations):
            for sample in range(num_samples):
                df = strong_combinatoriality_variable(coupling=perturbation, **kwds)
                df['sample'] = sample
                df['coupling'] = perturbation
                yield df
    return pd.concat(list(gen()))

def star_upto(V, K):
    for k in range(1, K+1):
        yield from s.cartesian_indices(V, k)

def strong_combinatoriality_variable(
        num_morphemes=4,
        morpheme_rate=.5,
        morpheme_length=1,
        maxlen=10,
        vocab_size=4,
        with_delimiter='both',
        coupling=0,
        coupling_type='product',
        source='zipf',
        debug=False,
        shuffle=True,
        len_granularity=1,
        **kwds):
    if source == 'zipf':
        morpheme_source = s.zipf_mandelbrot(num_morphemes, **kwds)
        if shuffle:
            np.random.shuffle(morpheme_source)
    elif source == 'rem':
        morpheme_source = s.rem(num_morphemes, **kwds)

    meanings = list(star_upto(num_morphemes, maxlen))
    unnormalized_source = np.array([
        morpheme_rate**len(m) * (1 - morpheme_rate) * np.prod(morpheme_source[list(m)])
        for m in meanings
    ])
    source = unnormalized_source / unnormalized_source.sum()
    if coupling:
        if source=='zipf':
            joint_source = s.zipf_mandelbrot(len(meanings), **kwds)
            if shuffle:
                np.random.shuffle(joint_source)
        else:
            joint_source = s.rem(len(meanings), **kwds)
        
        new_source = s.couple(source, joint_source, coupling=coupling, coupling_type=coupling_type)
        tc = new_source @ (np.log(new_source) - np.log(source))
        source = new_source
    else:
        tc = 0
    
    # Strongly systematic mapping from a morpheme to a "word"
    strong_code = c.random_code(num_morphemes, vocab_size, morpheme_length, unique=True)
    shuffles = random.sample(list(itertools.permutations(range(num_morphemes))), maxlen)
    weak_codes = [strong_code[list(shuffle)] for shuffle in shuffles] # one code per morpheme position
    signals = {
        'strong': c.form_probabilities(
            source,
            meanings,
            c.systematic_code(c.as_code(strong_code)),
            with_delimiter=with_delimiter,
        ),
        'weak': c.form_probabilities(
            source,
            meanings,
            c.weakly_systematic_code(list(map(c.as_code, weak_codes))),
            with_delimiter=with_delimiter,
        ),
    }
    signals['nonsys'] = pd.DataFrame({
        'form': shuffled(signals['strong']['form']),
        'probability': source,
    })
    signals['nonsysl'] = pd.DataFrame({
        'form': shuffle_preserving_length(signals['strong']['form'], granularity=len_granularity),
        'probability': source,
    })
        
    def gen():
        for name, signal in signals.items():
            curves = il.curves_from_sequences(signal['form'], weights=signal['probability'])
            curves['type'] = name
            curves['tc'] = tc
            yield curves

    if debug:
        breakpoint()

    return pd.concat(list(gen()))

def strong_combinatoriality(num_morphemes=4,
                            num_parts=4,
                            morpheme_length=1,
                            vocab_size=4,
                            with_delimiter='both',
                            coupling=0,
                            coupling_type='product',
                            source='zipf',
                            debug=False,
                            include_positional=False,
                            shuffle=True,
                            **kwds):
    # Defaults are set to remove redundancy.
    # Considering distributions over k variables, p(x_1, ..., x_k), which approximately
    # factorizes as iid p(x_1)...p(x_k).
    # Then it is better to be strongly-systematic, because then your unigram distribution
    # is minimal entropy (minimally mixing the iid source components).

    # BUT:
    # We only get strong < weak when with_delimiter='left', or when considering M/S tradeoff instead of EE
    # otherwise the lower h_1 is offset by higher h_t, because in the strongly-systematic code,
    # it's hard to tell where you are in the utterance; therefore the lower unigram entropy is offset
    # by higher bigram entropy for the end-of-sequence symbol.
   # Strong systematicity would only be better if time index is not informative about distance from end.

    # What distribution could show the strong-systematicity advantage using pure E?
    # -> Morpheme value needs to be uninformative about remaining utterance length.
    # This will only work if length is p(k) \propto e^-k.
    # But such a distribution cannot be represented in the current framework...
    num_meanings = num_morphemes**num_parts

    if source == 'zipf':
        morpheme_source = s.zipf_mandelbrot(num_morphemes, **kwds)
        if shuffle:
            np.random.shuffle(morpheme_source) 
    elif source == 'rem':
        morpheme_source = s.rem(num_morphemes, **kwds)
    joint_source = s.zipf_mandelbrot(num_meanings, **kwds)
    if shuffle:
        np.random.shuffle(joint_source)
        
    factored_source = np.array([1])
    for k in range(num_parts):
        factored_source = s.product_distro(factored_source, morpheme_source)
    if coupling:
        source = s.couple(factored_source, joint_source, coupling=coupling, coupling_type=coupling_type)
    else:
        source = factored_source
    source = source.reshape((num_morphemes,)*num_parts)
    tc = s.tc(source)

    # Build codes
    strong_code = c.random_code(num_morphemes, vocab_size, morpheme_length, unique=True)
    holistic_code = c.random_code(num_meanings, vocab_size, morpheme_length*num_parts, unique=True)

    # Weak systematic code is permutations of the strong code; this controls entropy rate
    shuffles = random.sample(list(itertools.permutations(range(num_morphemes))), num_parts)
    weak_codes = [strong_code[list(shuffle)] for shuffle in shuffles]

    # Free order lowers the entropy rate, why? It creates collisions. Avoid using positional coding?

    signals = dict([
        # Strongly systematic code:
        ('strong', c.word_probabilities(source, strong_code, with_delimiter=with_delimiter)),
        ('weak', c.word_probabilities(source, weak_codes, encode=c.encode_weak_contiguous, with_delimiter=with_delimiter)),
        ('holistic', c.word_probabilities(source.flatten(), holistic_code, with_delimiter=with_delimiter)),
    ])

    if include_positional:
        signals |=  dict([
            ('strong_positional', c.word_probabilities(source, strong_code, encode=c.encode_contiguous_positional, with_delimiter=with_delimiter)),

            #('free_order', c.word_probabilities(source, strong_code, encode=c.encode_contiguous_random_order, with_delimiter=with_delimiter)),
            ('free_positional', c.word_probabilities(source, strong_code, encode=c.encode_contiguous_positional_random_order, with_delimiter=with_delimiter)),
        ])
        


    def gen():
        for name, signal in signals.items():
            curves = il.curves_from_sequences(signal['form'], weights=signal['probability'])
            curves['type'] = name
            curves['tc'] = tc
            yield curves

    if debug:
        breakpoint()

    return pd.concat(list(gen()))

def sample_dfs(f, num_samples=1):
    def gen():
        for i in tqdm.tqdm(range(num_samples)):
            df = f()
            df['sample'] = i
            yield df
    return pd.concat(list(gen()))

def combinatoriality(morpheme_length=1,
                     vocab_size=2,
                     with_delimiter='both',
                     source='zipf',
                     coupling=0,
                     shuffle_source=True,
                     shuffle_forms=True,
                     **kwds):
    """ Sweep through codes of different levels of holisticity vs. systematicity. """
    # findings
    # with zero redundancy and coupling=1, no advantage to any factorization
    # with zero redundnacy and coupling=0, factorization is good
    # with redundancy and coupling=1, factorization is good
    # with redundancy and coupling=0, factorization is wonderful
    if source == 'zipf':
        joint_source = s.zipf_mandelbrot(256, **kwds)
        morpheme_source = s.zipf_mandelbrot(2, **kwds)
        if shuffle_source:
            np.random.shuffle(joint_source)
            np.random.shuffle(morpheme_source)
    elif source == 'rem':
        joint_source = s.rem(256, **kwds)
        morpheme_source = s.rem(2, **kwds)

    if coupling == 1:
        source = joint_source
    else:
        factored_source = np.array([1])
        for k in range(8):
            factored_source = s.product_distro(factored_source, morpheme_source)
        source = s.couple(factored_source, joint_source, coupling=coupling)
        
    # 256 = 4 x 4 x 4 x 4
    codes = [
        (2, 8, c.random_code(2, vocab_size, morpheme_length*1, unique=True)),     # 2^8
        (4, 4, c.random_code(4, vocab_size, morpheme_length*2, unique=True)),     # 4^4
        (16, 2, c.random_code(16, vocab_size, morpheme_length*4, unique=True)),   # 16**2
        (256, 1, c.random_code(256, vocab_size, morpheme_length*8, unique=True)), # 256**1
    ]
    def gen():
        for num_morphemes, num_parts, code in codes:
            shape = (num_morphemes,)*num_parts
            signal = c.word_probabilities(
                source.reshape(shape),
                code,
                with_delimiter=with_delimiter
            )
            curves = il.curves_from_sequences(signal['form'], weights=signal['probability'])
            curves['num_morphemes'] = num_morphemes
            curves['num_parts'] = num_parts
            curves['type'] = str(num_parts)
            curves['order'] = 'concatenative'
            yield curves

            if shuffle_forms:
                ds = sh.DeterministicScramble()
                shuffled_forms = signal['form'].map(ds.shuffle)
                curves = il.curves_from_sequences(shuffled_forms, weights=signal['probability'])
                curves['num_morphemes'] = num_morphemes
                curves['num_parts'] = num_parts
                curves['order'] = 'shuffle'
                curves['type'] = str(num_parts)
                yield curves
                
    df = pd.concat(list(gen()))
    df['tc'] = s.tc(source.reshape((2,)*8))
    return df
        
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
    df1 = summarize(source, id)
    df1['type'] = 'id'
    df2 = summarize(source, cnot)
    df2['type'] = 'cnot'
    return pd.concat([df1, df2])

def id_vs_cnot_sources(num_samples=1000, source='rem', redundancy=1, metric=il.ee, **kwds):
    id = np.repeat(np.array([
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    ]), redundancy, -1)
    cnot12 = np.repeat(np.array([
        [0, 2],
        [0, 3],
        [1, 3],
        [1, 2],
    ]), redundancy, -1)

    cnot21 = np.repeat(np.array([
        [0, 2],
        [1, 3],
        [1, 2],
        [0, 3],
    ]), redundancy, -1)

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
                'id': metric(summarize(source_flat, id, **kwds)),
                'cnot12': metric(summarize(source_flat, cnot12, **kwds)),
                'cnot21': metric(summarize(source_flat, cnot21, **kwds)),
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
        df = summarize(source, lang, **kwds)
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

def systematic_columns(code):
    patterns = {
        (0,0,0,0,1,1,1,1),
        (0,0,1,1,0,0,1,1),
        (0,1,0,1,0,1,0,1),
    }
    def recode(xs):
        i = 0
        seen = {}
        for x in xs:
            if x in seen:
                yield seen[x]
            else:
                seen[x] = i
                yield seen[x]
                i += 1
    return sum(tuple(recode(code[:,i])) in patterns for i in range(code.shape[-1]))

def three_sweep(i12=0, i23=0, p0=2/3, redundancy=1, **kwds):
    """ Sweep through all 2^3!=40320 unambiguous positional codes for a 3-bit source. """
    # p0 argument only used if i12=i23=0
    if not i12 and not i23:
        assert p0 <= .9
        source = s.product_distro(s.flip(p0+.1), s.product_distro(s.flip(p0+.05), s.flip(p0)))
    else:
        source = s.mi_mix3(i12, i23)
    id_code = np.repeat(np.array([
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 4],
        [0, 3, 5],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 4],
        [1, 3, 5],
    ]), redundancy, -1)
    permutations = list(itertools.permutations(range(8)))
    def gen():
        for permutation in tqdm.tqdm(permutations):
            code = id_code[list(permutation)]
            curves = summarize(source, code, **kwds)
            #curves['permutation'] = tuple(permutation)
            curves['code'] = str(code)
            curves['systematic'] = systematic_columns(code)
            curves['ee'] = il.ee(curves)
            curves['ms_auc'] = il.ms_auc(curves)
            yield curves
    return pd.concat(list(gen()))


# in the strong systematicity sweep, we're seeing an advantage for systematicity
# when the input bits are independent. but here, we're seeing no advantage for
# systematicity. why?

def id_vs_cnot3(i23=0.9, p0=.5, redundancy=1, make_plot=False, **kwds):
    source = s.product_distro(s.flip(p0 + .15), s.flip(p0))
    if i23:
        joint = np.array([1, 0, 0, 2]) / 3
        new_source = i23 * joint + (1-i23) * source
        mi = s.mi(new_source.reshape(2,2))
        source = new_source
    else:
        mi = 0
    source = s.product_distro(s.flip(p0 + .3), source)
    # I_23 = 0.5
    #source = s.mi_mix3(i12, i23)
    
    id_code = np.repeat(np.array([
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 4],
        [0, 3, 5],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 4],
        [1, 3, 5],
    ]), redundancy, -1)

    fuse12 = np.repeat(np.array([  # CNOT(1,2)
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 4],
        [0, 3, 5],
        [1, 3, 4],
        [1, 3, 5],
        [1, 2, 4],
        [1, 2, 5],        
    ]), redundancy, -1)

    fuse23 = np.repeat(np.array([  # CNOT(2,3)
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 5],
        [0, 3, 4],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 5],
        [1, 3, 4],        
    ]), redundancy, -1)

    fuse23_nonlocal = np.repeat(np.array([
        [2, 0, 4],
        [2, 0, 5],
        [3, 0, 5],
        [3, 0, 4],
        [2, 1, 4],
        [2, 1, 5],
        [3, 1, 5],
        [3, 1, 4],                
    ]), redundancy, -1)

    holistic = np.array(shuffled(id_code))
        

    df1 = summarize(source, id_code, **kwds)
    df1['type'] = 'id'
    df2 = summarize(source, fuse12, **kwds)
    df2['type'] = 'cnot12'
    df3 = summarize(source, fuse23, **kwds)
    df3['type'] = 'cnot23'
    df4 = summarize(source, holistic, **kwds)
    df4['type'] = 'holistic'

    df = pd.concat([df1, df2, df3, df4])
    df['mi'] = mi

    if make_plot:
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
    else:
        return df
    



def huffman_systematic_vs_not(first_prob=1/3, **kwds):
    # Meaning has two components, M_1 x M_2, with zero MI
    # It's just a product of an independent (1/3,2/3) * (1/2,1/4,1/8,1/8)
    # There are multiple legitimate Huffman codes for this source, one systematic and one not.
    # The systematic Huffman code has a better MS tradeoff due to less long-term dependencies -- conditional on delimiters. Why?
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
    df1 = summarize(source, systematic_huffman, **kwds)
    df1['type'] = 'systematic'
    df2 = summarize(source, nonsystematic_huffman, **kwds)
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
    df1 = summarize(source, compressed_code)
    df1['type'] = 'compressed'
    df2 = summarize(source, separable_local_code)
    df2['type'] = 'separable_local'
    df3 = summarize(source, separable_nonlocal_code)
    df3['type'] = 'separable_nonlocal'
    return pd.concat([df1, df2, df3])

def shuffled(xs):
    xs = list(xs)
    random.shuffle(xs)
    return xs

def locality(V=5, S=2, l=5, with_delimiter='both', **kwds):
    """ compare orders for {h, d1, d2} where I[h:d1] > I[h:d2] """
    # dhd is consistently best with S=20, l=5: unambiguous word boundaries.
    # with S=2, l=5 and left delimiter, dhd or hdd is the best, inconsistently.
    # -> The hdd advantage has something to do with phonotactics...
    # we rarely get consistent hdd advantage over hdh, but hdd_bad is always worse.
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

def mansfield_kemp(S=50, l=1, A_rate=1, N_rate=1, D_rate=1, with_delimiter='both'):
    # in English GUM, 
    # amod = 7747 = 32.6%
    # nummod = 928 = 3.9%
    # det = 11209 = 47.2%
    # NOUN = 23726

    # Spanish AnCora
    # amod = 29%
    # nummod = 6%
    # det = 85%
    
    source, meanings = s.mansfield_kemp_source(A_rate=A_rate, N_rate=N_rate, D_rate=D_rate)
    systematic, codebook = c.random_systematic_code(meanings, S, l, unique=True)
    return np_order(source, meanings, systematic, with_delimiter=with_delimiter)

def empirical_np_order(filename="data/de_np.csv", with_delimiter='both', truncate=0, len_limit=1):
    # default len_limit=1 excludes singleton noun NPs
    
    # Empirical MIs (lemmatized, unlemmatized)...
    # German, N=758,024 or 781,304 (unlemmatized)
    # N-Adj = 3.6, 5.6
    # N-Num = 1.7, 2.0
    # N-Det = 0.9, 1.6
    
    # Czech
    # N-Adj = 4.7
    # N-Num = 1.6
    # N-Det = 1.2

    # How to reduce these to manageable sizes? What is a manageable size anyway?
    ps, meanings = s.empirical_source(
        filename,
        truncate=truncate,
        len_limit=len_limit,
        rename={'N': 'n', 'Adj': 'A', 'Num': 'N', 'Det': 'D'}
    )
    print("Loaded source.", file=sys.stderr)
    return np_order(ps, meanings, c.identity_code, with_delimiter=with_delimiter, parts='nAND')

def np_order(source, meanings, code=c.identity_code, with_delimiter='both', parts="nAND"):    
    def codes():
        for order in itertools.permutations(range(len(parts))):
            label = "".join(parts[i] for i in order)
            grammar = [
                sorted(meaning, key=lambda x: label.index(x[0]))
                for meaning in meanings
            ]
            yield label, grammar
    the_codes = dict(codes())
    def gen():
        for name, order in tqdm.tqdm(the_codes.items()):
            forms = c.form_probabilities(source, order, code, with_delimiter=with_delimiter)
            curves = il.curves_from_sequences(forms['form'], forms['p'])
            curves['type'] = name
            yield curves
        return # skip the below; not informative
        # nonsystematic code, following the last permutation (DNAn, a good one)
        the_forms = map(code, order)
        df = pd.DataFrame({'form': the_forms, 'p': forms['p']})
        df['form'] = np.random.permutation(df['form'])
        # for utterance x, q(x) \propto |x| p(x), to create a true stationary distribution.
        df['p'] = df['p'] * (df['form'].map(len) + 1) # TODO: +1?
        Z = df['p'].sum()
        df['p'] = df['p'] / Z
        if with_delimiter == 'left':
            df['form'] = il.DELIMITER + df['form']
        elif with_delimiter:
            df['form'] = il.DELIMITER + df['form'] + il.DELIMITER
        curves = il.curves_from_sequences(df['form'], df['p'])
        curves['type'] = 'nonsys'
        yield curves

    # empirical typological frequencies from Dryer 2017
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
    typology['group'] = typology['type'].map(lambda t: first(s for s in groups if t in s))
    af = typology[['group', 'af', 'num_genera']].drop_duplicates().groupby('group').sum().reset_index()
    af.columns = ['group', 'af_sum', 'num_genera_sum']
    return pd.concat(list(gen())), typology.merge(af)

def first(xs):
    return next(iter(xs))

def plot_np_order(df, typology, depvar='af_sum', **kwds):
    df = typology.merge(df[df['t']==df['t'].max()])[['af_sum', 'num_genera_sum', 'group', 'H_M_lower_bound', 'h_t']]
    df['group_name'] = df['group'].map(lambda s: "/".join(sorted(s)))
    return df
            
def aanaa(S=50, l=1, with_delimiter='both'):
    # NA* with up to 4 A's in 3 classes, number of A's is geometric
    
    
    # For A*N sequences, 
    # 1. Class-level information locality: higher mi adjective classes go closer. -- Yes, if MI differential is big enough.
    # 2. Consistent AAN better than Gildea & Temperley inward-outward order. -- Yes, as long as with_delimiter='both' and p_halt > 0 --- consistent placement of N makes the boundary predictable.
    # 3. Class consistency is more important than individual pmi with noun. - Yes.
    # 4. Class-consistent ANA is better than G&T-style ANA. -- doesn't seem to work.

    # with 2-3 classes, each of which has some MI with the noun.
    # show that the code which orders by MI is best.
    # then show consistency is good -- an inconsistent inward-outward ordering is worse.


    # classes are defined by I[A:B|N] = 0. Conditionally independent given the N.

    # 5 adjectives, 5 nouns, 3 adjective classes:
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
    curves['type'] = 'consistent_local' # consistent local NAA
    curves2 = il.curves_from_sequences(forms_nonlocal['form'], forms_nonlocal['p'])
    curves2['type'] = 'consistent_nonlocal' # consistent nonlocal NAA
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


def frequent_fusion_irregularity(V, k, S=None, l=2, coupling=1, how_many=2, with_delimiter=True, **kwds):
    """ Minimizing E does *NOT* predict irregularity for frequent forms """
    # would it predict fusion for high PMI forms?
    assert how_many >= 1
    if S is None:
        S = V    
    source, meanings = s.mostly_independent(*(V,)*k, coupling=coupling, **kwds)
    codes = c.random_code(k*V, S, l, unique=True).reshape(k,V,l)
    holistic = c.random_code(V**k, S, l*k, unique=True)
    L = V**k
    def gen():
        forms = c.word_probabilities(source.reshape(*(V,)*k), codes, encode=c.encode_weak_contiguous, with_delimiter=with_delimiter).sort_values('probability', ascending=False, ignore_index=True)
        the_forms = forms['form']
        curves = il.curves_from_sequences(the_forms, forms['probability'])
        curves['type'] = 'systematic'
        yield curves

        # Swap top 2 forms to simulate fusion/irregularity
        forms_irrtop = pd.concat([the_forms[:how_many][::-1], the_forms[how_many:]])
        curves = il.curves_from_sequences(forms_irrtop, forms['probability'])
        curves['type'] = 'irrtop'
        yield curves

        forms_irrbottom = pd.concat([the_forms[:-how_many], the_forms[-how_many:][::-1]])
        curves = il.curves_from_sequences(forms_irrbottom, forms['probability'])
        curves['type'] = 'irrbottom'
        yield curves

        hol_forms = c.form_probabilities_np(source, holistic, with_delimiter=with_delimiter)['form']
        curves = il.curves_from_sequences(hol_forms, forms['probability'])
        curves['type'] = 'holistic'
        yield curves

        hybrid_top = pd.concat([hol_forms[:how_many], the_forms[how_many:]])
        curves = il.curves_from_sequences(hybrid_top, forms['probability'])
        curves['type'] = 'hybrid_top'
        yield curves

        hybrid_bottom = pd.concat([the_forms[:-how_many], hol_forms[-how_many:]])
        curves = il.curves_from_sequences(hybrid_bottom, forms['probability'])
        curves['type'] = 'hybrid_bottom'
        yield curves

    return pd.concat(list(gen())), codes, holistic

def paradigmatic32(redundancy=1, coupling=0, with_delimiter=True, **kwds):
    """ compare natural vs unnatural paradigm shapes. works with redundancy > 1 """
    # kind of works with redundancy=3, but different entropy rates
    source, meanings = s.mostly_independent(3, 2, coupling=coupling, **kwds) # 3x2
    paradigmatic_mergehigh = np.repeat(np.array([
        [0, 1], [0, 1], [0, 2],
        [3, 4], [3, 4], [3, 5]
    ]), redundancy, -1)
    paradigmatic_mergelow = np.repeat(np.array([
        [0, 1], [0, 2], [0, 2],
        [3, 4], [3, 5], [3, 5]
    ]), redundancy, -1)    
    inconsistent = np.repeat(np.array([
        [0, 1], [0, 1], [0, 2],
        [3, 4], [3, 5], [3, 5]
    ]), redundancy, -1)
    nonconvex = np.repeat(np.array([
        [0, 1], [0, 2], [0, 1],
        [3, 4], [3, 5], [3, 4]
    ]), redundancy, -1)
    
    def gen():
        forms = c.form_probabilities_np(source.flatten(), paradigmatic_mergehigh, with_delimiter=with_delimiter)
        curves = il.curves_from_sequences(forms['form'], forms['probability'])
        curves['type'] = 'product_high'
        yield curves

        forms = c.form_probabilities_np(source.flatten(), paradigmatic_mergelow, with_delimiter=with_delimiter)
        curves = il.curves_from_sequences(forms['form'], forms['probability'])
        curves['type'] = 'product_low'
        yield curves        

        forms2 = c.form_probabilities_np(source.flatten(), inconsistent, with_delimiter=with_delimiter)
        curves = il.curves_from_sequences(forms2['form'], forms2['probability'])
        curves['type'] = 'inconsistent'
        yield curves

        forms2 = c.form_probabilities_np(source.flatten(), nonconvex, with_delimiter=with_delimiter)
        curves = il.curves_from_sequences(forms2['form'], forms2['probability'])
        curves['type'] = 'nonconvex'
        yield curves        
    return pd.concat(list(gen()))

def paradigm_size_effect(A_min=2, A_max=10, B=2, S=None, l=1, with_delimiter=True, num_samples=100, coupling=.1, **kwds):
    """ Works with low coupling. holistic increases with size; systematic stays constant. """
    if S is None:
        S = A_max * B
    systematic_A = c.random_code(A_max, S, l, unique=True)
    systematic_B = c.random_code(B, S, l, unique=True)
    # need to control entropy rate at each A...
    def gen():
        for A in tqdm.tqdm(range(A_min, A_max + 1)):
            source, meanings = s.mostly_independent(A, B, coupling=coupling, **kwds)
            systematic = c.word_probabilities(
                source,
                [systematic_A, systematic_B],
                encode=c.encode_weak_contiguous,
                with_delimiter=with_delimiter
            )
            curves = il.curves_from_sequences(systematic['form'], systematic['probability'])
            curves['type'] = 'systematic'
            curves['A'] = A
            curves['sample'] = 0
            yield curves[curves['t'] == curves['t'].max()]

            for i in range(num_samples):

                # make holistic equivalent by permuting the systematic code
                systematic['holistic'] = np.random.permutation(systematic['form'])
                curves = il.curves_from_sequences(systematic['holistic'], systematic['probability'])
                curves['type'] = 'holistic'
                curves['sample'] = i + 1
                curves['A'] = A
                yield curves[curves['t'] == curves['t'].max()]

    return pd.concat(list(gen()))

def paradigmatic_holistic(A=3, B=2, num_words=4, S=4, l=1, with_delimiter=True, **kwds):
    """ Try all the ways of encoding an AxB paradigm with w holistic words. """
    source, meanings = s.mostly_independent(A, B, **kwds)
    code = c.random_code(num_words, S, l, unique=True)
    paradigms = c.paradigms(A*B, num_words)
    def gen():
        for paradigm in paradigms:
            if not il.is_monotonically_increasing(paradigm):
                ptype = 'nonconvex'
            elif False: # TODO
                ptype = 'inconsistent'
            else:
                ptype = 'product'
            paradigm_code = np.array([code[i] for i in paradigm])
            probs = c.form_probabilities_np(source.flatten(), paradigm_code, with_delimiter=with_delimiter)
            curves = il.curves_from_sequences(probs['form'], probs['probability'])
            curves['type'] = ptype
            curves['paradigm'] = "".join(map(str, paradigm))
            yield curves
    return pd.concat(list(gen()))

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

def gen_df(data):
    for name, i, df in data:
        df['type'] = name
        df['sample'] = i
        yield df

def word_level_mi(pair_counts, num_baseline_samples=1000):
    """ Take word pairs from a corpus and compare their MI to shuffles """

    def word_level():
        forms, weights = list(pair_counts.keys()), list(pair_counts.values())
        yield 'real', 0, pd.DataFrame({'mi': [mi_from_pair_counts(pair_counts.keys(), pair_counts.values())]})
        for i in range(num_baseline_samples):
            yield 'nonsys', i, pd.DataFrame({'mi': [mi_from_pair_counts(shuffled(pair_counts.keys()), pair_counts.values())]})

    return pd.concat(list(gen_df(word_level())))            


def dep_word_pairs(num_baseline_samples=1000,
                   len_granularity=1,
                   with_space=True,
                   with_delimiter='both',
                   keep_order=True,
                   require_adjacent=True,
                   **kwds):
    counts = f.raw_word_pair_counts( # English VO dependencies by default
        keep_order=keep_order,
        require_adjacent=require_adjacent,
        **kwds
    ) 
    return (
        word_level_mi(counts, num_baseline_samples=num_baseline_samples),
        letter_level(
            counts,
            num_baseline_samples=num_baseline_samples,
            len_granularity=len_granularity,
            with_space=with_space,
            with_delimiter=with_delimiter
        ),
    )

def shuffle_preserving_length(forms: pd.Series, granularity=1):
    lenclass = forms.map(len) // granularity
    assert il.is_monotonically_increasing(lenclass)
    new_forms = []
    for length in lenclass.drop_duplicates(): # ascending order
        mask = lenclass == length
        shuffled_forms = np.random.permutation(forms[mask])
        new_forms.extend(shuffled_forms)
    return new_forms

def ngrams(n=2):
    # start with strong n-gram models for 1:n, maybe neural
    # use them to calculate cross-entropy in a test set, hence E
    # compare to standard shuffles
    pass

def letter_level(counts, num_baseline_samples=1000, len_granularity=1, with_space=True, with_delimiter='both'):

    def format_form(xs):
        return "".join([
            il.DELIMITER if with_delimiter else "",
            (" " if with_space else "").join(xs),
            il.DELIMITER if with_delimiter == 'both' else ""
        ])

    def inner_letter_level():
        forms = list(map(format_form, counts.keys()))
        weights = list(counts.values())
        yield 'real', 0, il.curves_from_sequences(forms, weights)
        
        both = pd.DataFrame({'forms': forms, 'weights': weights})
        both['len'] = both['forms'].map(len)
        both = both.sort_values('len', ignore_index=True)
        forms, weights = both['forms'], both['weights']
        for i in tqdm.tqdm(range(num_baseline_samples)):
            yield 'nonsys', i, il.curves_from_sequences(np.random.permutation(forms), weights) # note: does not preserve entropy rate
            ds = sh.DeterministicScramble()
            # could form phonotactically ok-ish words using WOLEX?            
            yield 'dscramble', i, il.curves_from_sequences(map(ds.shuffle, forms), weights)
            shuffled_forms = shuffle_preserving_length(forms, granularity=len_granularity)
            yield 'nonsysl', i, il.curves_from_sequences(shuffled_forms, weights)

    return pd.concat(list(gen_df(inner_letter_level())))
        
# Ideas / Todo
# DONE. Correct strong systematicity study, or switch everything to MS tradeoff...
# 2. Adjective order with empirical frequencies across languages, and baselines as in AANAA
# 3. Agreement baselines: why is there agreement? why does it target certain dependencies andd not others? Idea: very low MI -> agreement bad; some MI -> some agreement good
#    1. Shuffle the agreement forms on adjectives/verbs -- grab a different form of the same lemma deterministically as a function of features. 
# 4. Word-level probability shuffles. Would this be counterevidence?







