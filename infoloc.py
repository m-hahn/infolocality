import sys
import random
import itertools
from collections import Counter
from typing import *

import tqdm
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats

EPSILON = 10 ** -8

def sliding_from_left(xs: Sequence, k: int) -> Iterator[Sequence]:
    """
    Sliding windows from the left of maximum size k.
    
    Example:
    >>> list(sliding_from_left("abcde", 3))
    ["a", "ab", "abc", "bcd", "cde"]
    
    """
    for i in range(len(xs)):
        yield xs[:i+1][-k:]

def curves_from_sequences(xs: Iterable[Sequence],
                          weights: Optional[Iterable]=None,
                          maxlen: Optional[int]=None,
                          monitor: bool=False) -> pd.DataFrame:
    counts = counts_from_sequences(xs, weights=weights, maxlen=maxlen, monitor=monitor)
    return curves_from_counts(counts, monitor=monitor)

def counts_from_sequences(xs: Iterable[Sequence],
                          weights: Optional[Iterable],
                          maxlen: Optional[int]=None,
                          monitor: bool=False) -> pd.DataFrame:
    """ Return a dataframe with columns t, x_{<t}, x_t, and count,
    where count is the weighted number of observations for the given
    x_{<t} followed by x_t in position t.
    """
    if maxlen is None:
        if not isinstance(xs, Sequence):
            xs = list(xs)
        maxlen = max(map(len, xs))
    if weights is None:
        weights = itertools.repeat(1)

    if monitor:
        print("Aggregating n-gram statistics...", file=sys.stderr)
    counts = Counter()
    for x, w in zip(tqdm.tqdm(xs, disable=not monitor), weights):
        # x is a string / sequence.
        # w is a weight / probability / count.
        for t in range(maxlen): # window size
            for chunk in sliding_from_left(x, t+1): 
                counts[t, chunk[:-1], chunk[-1]] += w
                
    df = pd.DataFrame(counts.keys())
    df.columns = ['t', 'x_{<t}', 'x_t']
    df['count'] = counts.values()
    return df

def curves_from_counts(df: pd.DataFrame, monitor: bool=False) -> pd.DataFrame:
    """ 
    Input: a dataframe with columns 't', 'x_{<t}' 'x_t', and 'count',
    where 'x_{<t}' gives a context, and 'count' gives a weight or count
    for an item in that context.
    """
    if monitor:
        print("Normalizing probabilities...", file=sys.stderr, end=" ")

    log_count = np.log(df['count'])

    Z_t = df.groupby('t')['count'].sum()
    assert np.allclose(Z_t[0], Z_t)
    joint_logp = log_count - np.log(Z_t[0])
    
    Z_context = df.groupby(['t', 'x_{<t}'])['count'].transform('sum')
    conditional_logp = log_count - np.log(Z_context)
    
    if monitor:
        print("Done.", file=sys.stderr)
        
    return curves(df['t'], joint_logp, conditional_logp)

def curves(t: Sequence,
           joint_logp: pd.Series,
           conditional_logp: pd.Series) -> pd.DataFrame:
    """ 
    Input:
    t: A vector of dimension D giving time indices for observations.
    joint_logp: A vector of dimension D giving joint probabilities for observations of x and context c.
    conditional_logp: A vector of dimension D giving conditional probabiltiies for observations of x given c.

    Output:
    A dataframe of dimension max(t)+1, with columns t, h_t, I_t, and H_M_lower_bound.
    """
    p = np.exp(joint_logp)
    h_t = -(p * conditional_logp).groupby([t]).sum()
    var_h_t = ((p * conditional_logp**2) - (p * conditional_logp)**2).groupby([t]).sum()
    I_t = -h_t.diff()
    assert (I_t[1:] > -EPSILON).all()
    H_M_lower_bound = np.cumsum(I_t * I_t.index)
    H_M_lower_bound[0] = 0
    df = pd.DataFrame({
        't': np.arange(len(h_t)),
        'h_t': np.array(h_t),
        'var_h_t': np.array(var_h_t),
        'I_t': np.array(I_t),
        'H_M_lower_bound': np.array(H_M_lower_bound),
    })
    return df

def ee(curves: pd.DataFrame) -> float:
    return curves['H_M_lower_bound'].max()

def transient_information(curves: pd.DataFrame) -> float:
    """ Transient information from Crutchfield & Feldman (2003: §4C) """
    h = curves['h_t'].min()
    L = curves['t'] + 1
    return np.sum(L * (curves['h_t'] - h))

def ms_auc(curves: pd.DataFrame) -> float:
    """
    Area under the memory--surprisal trade-off curve.
    Only comparable when two curves have the same entropy rate.
    """
    h = curves['h_t'].min()
    d_t = curves['h_t'] - h
    return np.trapz(y=d_t, x=curves['H_M_lower_bound'])

def score(J: Callable[[pd.DataFrame], float],
          forms: Iterable[Sequence],
          weights: Optional[Iterable]=None,
          maxlen: Optional[int]=None) -> float:
    curves = curves_from_sequences(forms, weights=weights, maxlen=maxlen)
    return J(curves)

def test_curve_properties():
    """ Test invariance properties of the entropy rate curve. """
    def gen_string(T=10, V=5):
        length = random.choice(range(T))
        stuff = [random.choice(range(V)) for _ in range(length)] + ['#']
        return tuple(stuff)
    data = [gen_string() for _ in range(1000)]
    def reverse(xs):
        return xs[:-1][::-1] + ('#',)
    for i in range(10):
        w = scipy.special.softmax(np.random.randn(len(data)))
        one = curves_from_sequences(data, weights=w)
        two = curves_from_sequences(map(reverse, data), weights=w)
        assert np.allclose(one['h_t'], two['h_t'])
        assert np.allclose(one['I_t'][1:], two['I_t'][1:])
        assert np.allclose(one['H_M_lower_bound'], two['H_M_lower_bound'])

    # Two binary languages with identical curves
    # X_4 = X_1 + X_2 + X_3 (mod 2)
    one = curves_from_sequences(['aaa0', 'aab1', 'aba1', 'abb0', 'baa1', 'bab0', 'bba0', 'bbb1'])
    # X_4 = X_1
    two = curves_from_sequences(['aaa0', 'aab0', 'aba0', 'abb0', 'baa1', 'bab1', 'bba1', 'bbb1'])
    assert np.allclose(one['h_t'], two['h_t'])
    assert np.allclose(one['I_t'][1:], two['I_t'][1:])
    assert np.allclose(one['var_h_t'], two['var_h_t'])
    assert np.allclose(one['H_M_lower_bound'], two['H_M_lower_bound'])    

    def mark_position(seq):
        return tuple(1000*x + y for x, y in enumerate(seq[:-1])) + ('#',)

    fixed_length_data = [tuple(random.choice(range(5)) for _ in range(10)) + ('#',) for _ in range(1000)]
    for i in range(10):
        w = scipy.special.softmax(np.random.randn(len(fixed_length_data)))
        one = curves_from_sequences(fixed_length_data, weights=w)
        two = curves_from_sequences(map(mark_position, fixed_length_data), weights=w)
        assert np.allclose(one['h_t'].min(), two['h_t'].min())
        assert np.allclose(one['H_M_lower_bound'].max(), two['H_M_lower_bound'].max())

def test_ee():
    """ Test excess entropy calculation against analytical formulas. """
    assert np.allclose(ee(curves_from_sequences(["ac#", "bd#"])), np.log(3) + (1/3)*np.log(2))
    for i in range(10):
        # Compare against analytical formula E_2 = \log 3 + 1/3 I_{12}.
        p = scipy.special.softmax(i*np.random.randn(2,2))
        formula = E2(p)
        the_ee = ee(curves_from_sequences(["ac#", "ad#", "bc#", "bd#"], p.flatten()))
        assert np.allclose(the_ee, formula)
        
    assert np.allclose(ee(curves_from_sequences(["ace#", "adf#", "bce#", "bdf#"])), np.log(4) + (1/4)*np.log(2))
    assert np.allclose(ee(curves_from_sequences(["ace#", "ade#", "bcf#", "bdf#"])), np.log(4) + (1/4)*2*np.log(2))
    assert np.allclose(ee(curves_from_sequences(["ace#", "adf#", "bcf#", "bde#"])), np.log(4) + (1/4)*2*np.log(2))
    
    sequences = ["ace#", "acf#", "ade#", "adf#", "bce#", "bcf#", "bde#", "bdf#"]
    for i in range(10):
        # Compare against analytical formula E_3 = \log 4 + 1/4 (TC_{123} - I_{123} + I_{13}).
        p = scipy.special.softmax(i*np.random.randn(2,2,2))
        formula = E3(p)
        the_ee = ee(curves_from_sequences(sequences, p.flatten()))        
        assert np.allclose(the_ee, formula)

def E2(p: np.array) -> float:
    """ Excess entropy for delimited strings of fixed length 2 """
    p_x = p.sum(0)
    p_y = p.sum(1)
    mi = scipy.stats.entropy(p_x) + scipy.stats.entropy(p_y) - scipy.stats.entropy(p, axis=None)
    return np.log(3) + 1/3*mi

def E3(p: np.array) -> float:
    """ Excess entropy for delimited strings of fixed length 3 """
    p1 = p.sum(axis=(1,2))
    p2 = p.sum(axis=(0,2))        
    p3 = p.sum(axis=(0,1))
    p12 = p.sum(axis=2)
    p23 = p.sum(axis=0)        
    p13 = p.sum(axis=1)
    i13 = scipy.stats.entropy(p1) + scipy.stats.entropy(p3) - scipy.stats.entropy(p13, axis=None)
    tc = scipy.stats.entropy(p1) + scipy.stats.entropy(p2) + scipy.stats.entropy(p3) - scipy.stats.entropy(p, axis=None)
    i123 = (
        scipy.stats.entropy(p1) + scipy.stats.entropy(p2) + scipy.stats.entropy(p3)
        - scipy.stats.entropy(p12, axis=None) - scipy.stats.entropy(p23, axis=None) - scipy.stats.entropy(p13, axis=None)
        + scipy.stats.entropy(p, axis=None)
    )
    formula = np.log(4) + 1/4*(tc - i123 + i13)
    return formula

if __name__ == '__main__':
    import nose
    nose.runmodule()
 


