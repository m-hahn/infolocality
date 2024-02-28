import itertools
import operator
from math import log, exp
from collections import Counter
from typing import *

import tqdm
import numpy as np
import pandas as pd
import scipy.special

DELIMITER = '#'
EPSILON = 10 ** -5

def buildup(iterable):
    """ Build up

    Example:
    >>> list(buildup("abcd"))
    [('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd')]

    """
    so_far = []
    for x in iterable:
        so_far.append(x)
        yield tuple(so_far)

def sliding(iterable, n):
    """ Sliding

    Yield adjacent elements from an iterable in a sliding window
    of size n.

    Parameters:
        iterable: Any iterable.
        n: Window size, an integer.

    Yields:
        Tuples of size n.

    Example:
        >>> lst = ['a', 'b', 'c', 'd', 'e']
        >>> list(sliding(lst, 2))
        [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')]

    """
    its = itertools.tee(iterable, n)
    for i, iterator in enumerate(its):
        for _ in range(i):
            try:
                next(iterator)
            except StopIteration:
                return zip([])
    return zip(*its)

def is_monotonic(comparator, sequence, epsilon=EPSILON):
    def conditions():
        for x1, x2 in sliding(sequence, 2):
            yield comparator(x1, x2) or comparator(x1, x2+epsilon) or comparator(x1, x2-epsilon)
    return all(conditions())

def is_monotonically_decreasing(sequence, epsilon=EPSILON):
    return is_monotonic(operator.ge, sequence, epsilon=epsilon)

def is_monotonically_increasing(sequence, epsilon=EPSILON):
    return is_monotonic(operator.le, sequence, epsilon=epsilon)

def is_nonnegative(x, epsilon=EPSILON):
    return x + epsilon >= 0

def curves_from_sequences(xs, weights=None, labels=None, **kwds):
    counts = counts_from_sequences(xs, weights=weights, labels=labels, **kwds)
    if labels is None:
        label_values = []
    else:
        label_values = [counts[label] for label in labels.keys()]
    curves = curves_from_counts(counts['count'], counts['x_{<t}'], *label_values)
    return curves

def lattice_curves_from_sequences(xs, labels, weights=None, **kwds):
    # first get curves of h_t
    h = curves_from_sequences(xs, weights=weights, **kwds)
    
    # then get curves of h_t given each individual g
    hg = curves_from_sequences(xs, labels=labels, weights=weights, **kwds)

    # now average over the g -- ASSUMES EACH FORM HAS A UNIQUE LABEL
    Z = weights.sum()
    label_weights = pd.DataFrame(labels | {'weight': weights})
    label_weights['weight'] = label_weights['weight'] / label_weights['weight'].sum() # normalize
    hg = pd.merge(hg, label_weights)
    hg['h_t'] = hg['weight'] * hg['h_t']
    hG = hg[['t', 'h_t']].groupby(['t']).mean().reset_index()
    hG.columns = ['t', 'h_t_g']
    return pd.merge(h, hG)

def curves_from_counts(counts, context, *labels):
    """ 
    Input: counts, a dataframe with columns 'x_{<t}' and 'count',
    where 'x_{<t}' gives a context, and count gives a weight or count
    for an item in that context.
    """ 
    t = context.map(len)
    the_joint_logp = conditional_logp(counts, t, *labels)
    the_conditional_logp = conditional_logp(counts, context, *labels)
    return curves(t, the_joint_logp, the_conditional_logp, *labels) # TODO how to labels fit in here?

def counts_from_sequences(xs: Iterable[Sequence],
                          weights: Optional[Iterable],
                          labels=None,
                          maxlen=None,
                          monitor=False):
    """ Return a dataframe with column x_{<t}, x_t, and count,
    where count is the weighted number of observations for the given x_{<t} followed by x_t.
    """
    if maxlen is None:
        if not isinstance(xs, Sequence):
            xs = list(xs)
        maxlen = max(map(len, xs))
    if weights is None:
        weights = itertools.repeat(1)
    if labels is None:
        labels = {}
        
    counts = Counter()
    for x, w, *l in zip(tqdm.tqdm(xs, disable=not monitor), weights, *labels.values()):
        # x is a string/sequence.
        # w is a weight / probability / count.
        # l is a sequence of label values.
        for context, x_t in thing_in_context(x):
            for subcontext in padded_subcontexts(context, maxlen):
                counts[(*l, subcontext, x_t)] += w

    df = pd.DataFrame(counts.keys()) 
    df.columns = list(labels.keys()) + ['x_{<t}', 'x_t']
    df['count'] = counts.values()
    return df

def conditional_logp(counts, *contexts):
    if not contexts:
        Z = counts.sum()
        return np.log(counts) - np.log(Z)
    else:
        df =  pd.DataFrame({'count': counts})
        for i, context in enumerate(contexts):
            df[i] = context
        context_cols = list(range(len(contexts)))
        Z_context = df.groupby(context_cols).sum().reset_index() # this is slow...
        Z_context.columns = [*context_cols, 'Z']
        # big blowup here:
        df = df.join(Z_context.set_index(context_cols), on=context_cols) # preserve order
        return np.log(counts) - np.log(df['Z'])

def increments(xs):
    for *context, x in buildup(xs):
        yield tuple(context), x

def incremental_logp(x, logp):
    """ Only for all sequences of the same length! x must be a string! """
    def gen():
        for form, logp_form in zip(x, logp):
            logT = log(len(form))
            for context, x_t in increments(form):
                yield context, x_t, logp_form

    full = pd.DataFrame(data=gen())
    full.columns = ['x_{<t}', 'x_t', 'joint_logp']
    
    marg = full.groupby(['x_{<t}', 'x_t']).aggregate(scipy.special.logsumexp).reset_index()
    t = full['x_{<t}'].map(len)
    context = marg[['x_{<t}', 'joint_logp']].groupby(['x_{<t}']).aggregate(scipy.special.logsumexp).reset_index()
    context.columns = ['x_{<t}', 'context_logp']
    
    d = pd.merge(marg, context)
    d['conditional_logp'] = d['joint_logp'] - d['context_logp']
    return d

def curves(t, joint_logp, conditional_logp, *labels):
    """ 
    Input:
    t: A vector of dimension D giving time indices for observations.
    joint_logp: A vector of dimension D giving joint probabilities for observations of x and context c.
    conditional_logp: A vector of dimension D giving conditional probabiltiies for observations of x given c.

    Output:
    A dataframe of dimension max(t)+1, with columns t, h_t, I_t, and H_M_lower_bound.
    """
    if labels:
        return conditional_curves(t, joint_logp, conditional_logp, *labels)
    else:
        return unconditional_curves(t, joint_logp, conditional_logp)

def conditional_curves(t, joint_logp, conditional_logp, *labels):
    plogp = np.exp(joint_logp) * conditional_logp
    h_t = -plogp.groupby([t, *labels]).sum()
    feature_names = h_t.index.names[1:]
    t_name = h_t.index.names[0]
    assert h_t.groupby(feature_names).agg(is_monotonically_decreasing).all()
    I_t = -h_t.groupby(feature_names).diff().reset_index()
    I_t['tI_t'] = I_t[t_name] * I_t[0]
    H_M_lower_bound = I_t.groupby(feature_names).cumsum()['tI_t']
    H_M_lower_bound[0] = 0
    df = pd.DataFrame({
        't': np.array(I_t[t_name]),
        'h_t': np.array(h_t),
        'I_t': np.array(I_t[0]),
        'H_M_lower_bound': np.array(H_M_lower_bound),
    })
    for name in feature_names:
        df[name] = np.array(I_t[name])
    return df

def unconditional_curves(t, joint_logp, conditional_logp):    
    plogp = np.exp(joint_logp) * conditional_logp
    h_t = -plogp.groupby([t]).sum()
    assert is_monotonically_decreasing(h_t)
    I_t = -h_t.diff()
    H_M_lower_bound = np.cumsum(I_t * I_t.index)
    H_M_lower_bound[0] = 0
    df = pd.DataFrame({
        't': np.arange(len(h_t)),
        'h_t': np.array(h_t),
        'I_t': np.array(I_t),
        'H_M_lower_bound': np.array(H_M_lower_bound),
    })
    return df

def ee(curves):
    return curves['H_M_lower_bound'].max()

def ms_auc(curves):
    """
    Area under the memory--surprisal trade-off curve.
    Only comparable when two curves have the same entropy rate.
    """
    h = curves['h_t'].min()
    d_t = curves['h_t'] - h
    return np.trapz(y=d_t, x=curves['H_M_lower_bound'])

def score(J, forms, weights=None, maxlen=None):
    counts = counts_from_sequences(forms, weights=weights, maxlen=maxlen)
    curves = curves_from_counts(counts['count'], counts['x_{<t}'])
    return J(curves)

def rjust(xs, length, value):
    """ rjust for general iterables, not just strings """
    num_needed = length - len(xs)
    if num_needed <= 0:
        return xs
    else:
        r = [value] * num_needed
        r.extend(xs)
        return type(xs)(r)

def test_rjust():
    assert rjust("auc", 10, "#") == "#######auc"
    assert rjust("auc", 1, "#") == "auc"
    assert rjust(tuple("auc"), 10, '#') == tuple("#######auc")

def padded_subcontexts(context, maxlen):
    yield ""
    for length in range(1, maxlen): # need a +1?
        try:
            yield context[-length:].rjust(length, DELIMITER)
        except AttributeError:
            yield rjust(context[-length:], length, DELIMITER)

def restorer(xs):
    if isinstance(xs, str):
        return "".join
    else:
        return tuple            
            
def thing_in_context(xs: Sequence[Any]):
    restore = restorer(xs)
    if xs[0] is DELIMITER:
        context = [DELIMITER]
        xs = xs[1:]
    else:
        context = []
    for x in xs:
        yield restore(context), x
        context.append(x)
   
if __name__ == '__main__':
    import nose
    nose.runmodule()
 

