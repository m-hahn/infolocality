import itertools

import tqdm
import torch
import rfutils

EOS = "#"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
flat = itertools.chain.from_iterable

def half_sliding(xs, k):
    for chunk in rfutils.buildup(xs):
        yield chunk[-k:]

def increments(xs, k):
    for *context, x in half_sliding(xs, k):
        yield tuple(context), x

def how_many_hits(string, context, x, k):
    """ Return whether string starts with context+x. """
    chunk = context + (x,)
    return sum(window == chunk for window in half_sliding(string, k))

def entropy(p):
    return torch.special.entr(p).sum()

def conditional_entropy(p):
    return entropy(p) - entropy(p.sum(-1))

def unique(xs):
    seen = set()
    for x in xs:
        if x not in seen:
            yield x
            seen.add(x)
        else:
            seen.add(x)

class MarkovTransform:
    """ Keep track of the mapping from sequence indices to sequences of bounded length. """
    def __init__(self, X, k):
        self.X = X # for example, [0, 1, 00, 01, 10, 11].
        self.k = k # Markov order
        
        contexts, V = zip(*flat(increments(x, k) for x in self.X))
        self.contexts = {c:i for i, c in enumerate(unique(contexts))}
        self.V = {v:i for i, v in enumerate(unique(V))}
        # mask is a mapping XCV saying whether string CV is a prefix of string X
        self.mask = torch.zeros(len(self.X), len(self.contexts), len(self.V)).to(device)
        for i, x in enumerate(self.X):
            for context, v in increments(x, k):
                compatible = how_many_hits(x, context, v, k)
                self.mask[i, self.contexts[context], self.V[v]] = compatible
        self.marginal_mask = self.mask.sum(0).to(device)
        self.zeros = torch.zeros(len(self.contexts), len(self.V)).to(device)

    def transform(self, joint):
        """ Transform p(..., x) -> p(..., x_{<t}, x_t) """
        prefix = torch.einsum("...x,xcv->...cv", joint, self.mask)
        prefix = prefix / prefix.sum()
        return prefix.where(self.marginal_mask > 0, self.zeros) # stop gradient when mask gives probability zero


def ee(transforms, p_x):
    h_k = [
        conditional_entropy(transform.transform(p_x))
        for transform in transforms
    ]
    h = h_k[-1]
    k = transforms[-1].k
    return sum(h_k) - k*h

def lang_opt(source,
             V,
             k,
             mi_weight=1,
             ee_weight=1,
             nondeterminism_weight=0,
             positional=False,
             include_eos=False,
             init_beta=1,
             num_iter=100,
             print_every=10,
             debug=False,
             monitor=False,
             **kwds):
    if positional:
        strings = list(rfutils.cartesian_distinct_indices(V, k))
    else:
        strings = list(rfutils.cartesian_indices(V, k))
    if include_eos:
        strings = [s + (EOS,) for s in strings]
    transforms = [MarkovTransform(strings, t) for t in tqdm.tqdm(range(1, k+1+include_eos), disable=not monitor)]
    energy = (init_beta*torch.randn(len(source), len(strings))).clone().requires_grad_(True).to(device)
    opt = torch.optim.AdamW(params=[energy], **kwds)
    lnp_g = torch.log(source)[:, None] # shape G1
    for i in tqdm.tqdm(range(num_iter), disable=not monitor):
        opt.zero_grad()
        lnq_x_given_g = torch.log_softmax(energy, -1) # shape GX
        lnq_gx = lnp_g + lnq_x_given_g
        lnq_x = torch.logsumexp(lnq_gx, -2) # shape X
        E = ee(transforms, lnq_x.exp())
        H_x_given_g = -(torch.exp(lnq_gx) * lnq_x_given_g).sum()
        H_x = -torch.exp(lnq_x) @ lnq_x
        mi = H_x - H_x_given_g # at most H_x. 
        loss = ee_weight*E - mi_weight*mi + nondeterminism_weight*H_x_given_g
        if i % print_every == 0:
            print(f"i={i}, J={loss}, EE={E}, MI={mi}, H={H_x_given_g}")
        if debug:
            breakpoint()
        loss.backward()            
        opt.step()
    return torch.log_softmax(energy, -1).detach()

SHAPES = ['circle', 'triangle', 'square', 'rectangle']
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
COLOR_SHAPE_SOURCE = torch.softmax(torch.Tensor([ # propert shape 6x4
    -1.5040, -2.1875, -3.2165, -2.3789, -8.2091, -7.9948, -9.5170, -8.5365,
    -2.8550, -2.7071, -4.0052, -3.0344, -2.9795, -3.5183, -4.5010, -3.1453,
    -2.5130, -3.4727, -4.1353, -3.1229, -4.4349, -4.1126, -6.0600, -5.5061
]), -1)
COLOR_SHAPES = [
    'red circle',
    'red triangle',
    'red square',
    'red rectangle',
    'orange circle',
    'orange triangle',
    'orange square',
    'orange rectangle',
    'yellow circle',
    'yellow triangle',
    'yellow square',
    'yellow rectangle',
    'green circle',
    'green triangle',
    'green square',
    'green rectangle',
    'blue circle',
    'blue triangle',
    'blue square',
    'blue rectangle',
    'purple circle',
    'purple triangle',
    'purple square',
    'purple rectangle'
]


# 02 20 
