import itertools

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
             **kwds):
    if positional:
        strings = list(rfutils.cartesian_distinct_indices(V, k))
    else:
        strings = list(rfutils.cartesian_indices(V, k))
    if include_eos:
        strings = [s + (EOS,) for s in strings]
    transforms = [MarkovTransform(strings, t) for t in range(1, k+1+include_eos)]
    energy = (init_beta*torch.randn(len(source), len(strings))).clone().requires_grad_(True).to(device)
    opt = torch.optim.AdamW(params=[energy], **kwds)
    for i in range(num_iter):
        opt.zero_grad()
        q_gx = torch.softmax(energy, -1) # shape GX
        q_x = source @ q_gx
        E = ee(transforms, q_x)
        H_x_given_g = -torch.xlogy(source[:, None] * q_gx, q_gx).sum()
        H_x = -torch.xlogy(q_x, q_x).sum()
        mi = H_x - H_x_given_g # at most H_x. 
        loss = ee_weight*E - mi_weight*mi + nondeterminism_weight*H_x_given_g
        if i % print_every == 0:
            print(f"i={i}, J={loss}, EE={E}, MI={mi}, H={H_x_given_g}")
        if debug:
            breakpoint()
        loss.backward()            
        opt.step()
    return torch.softmax(energy, -1).detach()
        
    
    
