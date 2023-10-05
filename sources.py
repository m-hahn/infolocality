import itertools
from collections import Counter

import numpy as np
import scipy.special
import pandas as pd
import einops

MK_PATH = "data/generated-MI-distros/generated-MI-distros_%s.txt"

def log_rem(*shape, T=1):
    return scipy.special.log_softmax(1/T*np.random.randn(*shape))

def rem(*shape, T=1):
    return scipy.special.softmax(1/T*np.random.randn(*shape))

def zipf_mandelbrot(N, s=1, q=0):
    k = np.arange(N) + 1
    p = 1/(k+q)**s
    Z = p.sum()
    return p/Z

def log_zipf_mandelbrot(N, s=1, q=0):
    k = np.arange(N) + 1
    energy = -s * np.log(k+q)
    lnZ = scipy.special.logsumexp(energy)
    return energy - lnZ

def factor(p, m, k):
    return p.reshape(*(m,)*k)

def product_distro(p1, p2):
    return np.outer(p1, p2).flatten()

def log_product_distro(lp1, lp2):
    return lp1[:, None] + lp2[None, :]

def flip(p):
    return np.array([p, 1-p])

def log_flip(p):
    return np.log(np.array([p, 1-p]))

def mi_mix(pi):
    """ Generate a 2x2 distribution with MI monotonically related to pi. """
    A = np.ones(4)/2
    A[0b00] *= pi*1 + (1-pi)*1/2  # 00
    A[0b01] *= pi*0 + (1-pi)*1/2  # 01
    A[0b10] *= pi*0 + (1-pi)*1/2  # 10
    A[0b11] *= pi*1 + (1-pi)*1/2  # 11
    return A

def mi_mix3(i12, i23):
    A = np.ones(8)/2 # start with 1/2 everywhere
    A[0b000] *= (i12*1 + (1-i12)*1/2) * (i23*1 + (1-i23)*1/2) 
    A[0b001] *= (i12*1 + (1-i12)*1/2) * (i23*0 + (1-i23)*1/2) 
    A[0b010] *= (i12*0 + (1-i12)*1/2) * (i23*0 + (1-i23)*1/2) 
    A[0b011] *= (i12*0 + (1-i12)*1/2) * (i23*1 + (1-i23)*1/2)
    A[0b100] *= (i12*0 + (1-i12)*1/2) * (i23*1 + (1-i23)*1/2) 
    A[0b101] *= (i12*0 + (1-i12)*1/2) * (i23*0 + (1-i23)*1/2) 
    A[0b110] *= (i12*1 + (1-i12)*1/2) * (i23*0 + (1-i23)*1/2) 
    A[0b111] *= (i12*1 + (1-i12)*1/2) * (i23*1 + (1-i23)*1/2)
    return A

def entropy(p):
    return scipy.special.entr(p).sum()

def conditional_entropy(p_xy, p_x):
    """ get H[Y | X] """
    p_x = p_xy.sum(axis=-1, keepdims=True)
    return scipy.special.entr(p_xy).sum() - scipy.special.entr(p_x).sum()

def mi(p_xy):
    """ Calculate mutual information of a distribution P(x,y) 

    Input: 
    p_xy: An X x Y array giving p(x,y)
    
    Output:
    The mutual information I[X:Y], a nonnegative scalar,
    """
    p_x = p_xy.sum(axis=-1, keepdims=True)
    p_y = p_xy.sum(axis=-2, keepdims=True)
    return scipy.special.entr(p_x).sum() + scipy.special.entr(p_y).sum() - scipy.special.entr(p_xy).sum()

def perturbed_conditional(px, py, perturbation_size=1):
    """
    p(y | x) \propto p(y) exp(perturbation_size * a(x,y)) where a(x,y) ~ Normal.
    Shape X x Y.
    """
    a = np.random.randn(len(px), len(py))
    energy = np.log(py)[None, :] + perturbation_size*a
    return scipy.special.softmax(energy, -1)

def dependents(V, k, coupling=1, source='rem', consistent=True, **kwds):
    if source == 'zipf':
        h = zipf_mandebelbrot(V, **kwds)
        ds = [zipf_mandelbrot(V, **kwds) for _ in range(k)]
    elif source == 'rem':
        h = rem(V, **kwds)
        if not consistent:
            ds = rem(k, V, **kwds)
        else:
            d = rem(V, **kwds)
            ds = np.array([d for _ in range(k)])
    conditionals = np.array([
        perturbed_conditional(h, d, perturbation_size=coupling*np.sqrt(V)/(i+1))
        for i, d in enumerate(ds)
    ])
    mis = [mi(h[:, None] * conditional) for conditional in conditionals]
    def gen():
        for configuration in itertools.product(*(range(V),)*(k+1)):
            d = {'h': ('h', configuration[0])}
            d |= {'d%d' % i : d for i, d in enumerate(configuration[1:])}
            p = h[configuration[0]]
            for i, dep in enumerate(configuration[1:]):
                p *= conditionals[i, configuration[0], dep]
            yield d, p
    meanings, ps = zip(*gen())
    return ps, meanings, mis

def astarn(num_A, num_N, num_classes=1, p_halt=.5, maxlen=4, source='rem', coupling=1, **kwds):
    # UGH! Needs to be a noun plus a MULTISET of adjectives, unordered...
    if source == 'zipf':
        As = [zipf_mandelbrot(num_A, **kwds) for _ in range(num_classes)]
        N = zipf_mandelbrot(num_N, **kwds)
    elif source == 'rem':
        As = rem(num_classes, num_A, **kwds)
        N = rem(num_N, **kwds)
    ANs = np.array([
        perturbed_conditional(N, A, perturbation_size=coupling*np.sqrt(num_A)/(k+1))
        for k, A in enumerate(As)
    ]) # shape C x N x A
    AN = N[None, :] * einops.rearrange(ANs, "c n a -> (c a) n") / num_classes # shape AN
    A = AN.sum(axis=-1, keepdims=True)
    pmi = einops.rearrange(np.log(AN) - np.log(A) - np.log(N), "(c a) n -> c n a", c=num_classes, a=num_A)
    class_mis = [mi(N[:, None] * AN) for AN in ANs]   
    def gen():
        for n in range(num_N):
            d = {'n': n, 'p': N[n]}
            for k in range(maxlen):
                p_k = (1-p_halt)**k * (p_halt if k <= maxlen else 1) / num_classes
                for classes in itertools.product(*(range(num_classes),)*k):
                    for adjectives in itertools.product(*(range(num_A),)*k):
                        d = {'n': (n,)}
                        d |= {str(i) : (c,a) for i, (c, a) in enumerate(zip(classes, adjectives))}
                        p = N[n] * p_k * np.prod([ANs[c,n,a] for c, a in zip(classes, adjectives)])
                        pmi_d = {str(i) : pmi[c,a,n] for i, (c,a) in enumerate(zip(classes, adjectives))}
                        yield p, d, pmi_d
    ps, meanings, pmis = zip(*gen())
    # want conditional entropy of nouns given each adjective
    ps = np.array(ps)
    ps = ps / np.sum(ps)
    return ps, meanings, pmis, class_mis

def mansfield_kemp_source(which='training'): # or 'test'
    # Noun Adj Num Dem
    filename = MK_PATH % which
    df = pd.read_csv(filename, sep="\t")
    counts = Counter(tuple(m.values()) for m in df.to_dict(orient='record'))
    meanings = list(counts.keys())
    ps = np.array(list(counts.values()))
    ps = ps / ps.sum()
    return ps, meanings
    






    
