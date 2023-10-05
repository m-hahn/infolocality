import itertools
import operator
from math import log, exp
from collections import Counter

import tqdm
import rfutils
import numpy as np
import pandas as pd
import scipy.special

DELIMITER = '#'
EPSILON = 10 ** -5

def is_monotonic(comparator, sequence, epsilon=EPSILON):
    def conditions():
        for x1, x2 in rfutils.sliding(sequence, 2):
            yield comparator(x1, x2) or comparator(x1, x2+epsilon) or comparator(x1, x2-epsilon)
    return all(conditions())

def is_monotonically_decreasing(sequence, epsilon=EPSILON):
    return is_monotonic(operator.ge, sequence, epsilon=epsilon)

def is_nonnegative(x, epsilon=EPSILON):
    return x + epsilon >= 0

def curves_from_sequences(xs, weights=None, maxlen=None, labels=None, monitor=False):
    counts = counts_from_sequences(xs, maxlen=maxlen, weights=weights, labels=labels, monitor=monitor)
    if labels is None:
        label_values = []
    else:
        label_values = [counts[label] for label in labels.keys()]
    curves = mle_curves_from_counts(counts['count'], counts['x_{<t}'], *label_values)
    return curves

def lattice_curves_from_sequences(xs, labels, maxlen=None):
    h = curves_from_sequences(xs, maxlen=maxlen)
    hg = curves_from_sequences(xs, maxlen=maxlen, labels=labels).groupby(['t']).mean().reset_index()
    hg.columns = "t h_t_G I_t_G H_M_G_lower_bound".split()
    return pd.merge(h, hg)

def mle_curves_from_counts(counts, context, *labels):
    """ 
    Input: counts, a dataframe with columns 'x_{<t}' and 'count',
    where 'x_{<t}' gives a context, and count gives a weight or count
    for an item in that context.
    """ 
    t = context.map(len)
    joint_logp = conditional_logp_mle(counts, t, *labels)
    conditional_logp = conditional_logp_mle(counts, context, *labels)
    return curves(t, joint_logp, conditional_logp, *labels) # TODO how to labels fit in here?

def counts_from_sequences(xs, weights=None, labels=None, maxlen=None, monitor=False):
    """ Return a dataframe with column x_{<t}, x_t, and count,
    where count is the number of observations for the given x_{<t} followed by x_t. """
    if maxlen is None:
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
        # l is a sequence of labels.
        for context, x_t in thing_in_context(x):
            for subcontext in padded_subcontexts(context, maxlen):
                counts[(*l, subcontext, x_t)] += w

    df = pd.DataFrame(counts.keys())
    df.columns = list(labels.keys()) + ['x_{<t}', 'x_t']
    df['count'] = counts.values()
    return df

def logp_mle(counts):
    Z = counts.sum()
    return np.log(counts) - np.log(Z)

def conditional_logp_mle(counts, *contexts):
    if not contexts:
        return logp_mle(counts)
    else:
        df =  pd.DataFrame({'count': counts})
        for i, context in enumerate(contexts):
            df[i] = context
        context_cols = list(range(len(contexts)))
        Z_context = df.groupby(context_cols).sum().reset_index() # this is slow...
        Z_context.columns = [*context_cols, 'Z']
        df = df.join(Z_context.set_index(context_cols), on=context_cols) # preserve order
        return np.log(counts) - np.log(df['Z'])

def increments(xs):
    for *context, x in rfutils.buildup(xs):
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
    I_t = (-h_t.groupby(feature_names).diff()).reset_index()
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
    """ Area under the memory--surprisal trade-off curve. """
    h = curves['h_t'].min()
    d_t = curves['h_t'] - h
    return np.trapz(y=d_t, x=curves['H_M_lower_bound'])

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
    for length in range(1, maxlen): # TODO: used to have a +1 here, why?
        try:
            yield context[-length:].rjust(length, DELIMITER)
        except AttributeError:
            yield rjust(context[-length:], length, DELIMITER)

def restorer(xs):
    if isinstance(xs, str):
        return "".join
    else:
        return tuple            
            
def thing_in_context(xs):
    restore = restorer(xs)
    if xs[0] is DELIMITER:
        context = [DELIMITER]
        xs = xs[1:]
    else:
        context = []
    for x in xs:
        yield restore(context), x
        context.append(x)

   
