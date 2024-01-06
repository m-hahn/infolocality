import sys
import itertools
from math import log, exp

import rfutils
import numpy as np
import scipy.special
import pandas as pd
import tqdm

import pmonad
import pcfg
import infoloc as il
import shuffles as sh

def demonstrate_systematic_learning_22(num_samples=1000, num_runs=1, **kwds):
    return learn_from_samples(K=2, V=2, num_samples=num_samples, targets=[
        0, # fully systematic
        1, # cnot
        7, # weakly systematic
    ]*num_runs, **kwd)

def demonstrate_systematic_learning_23(num_samples=20000, num_runs=1, **kwds):
    targets = [
        0, # strongly systematic
        1, # toffoli(1,2,3) -- flip on more frequent control bits
        5040, # toffoli(not(1), not(2), 3) -- flip on less frequent control bits
        121, # cnot(2,3) -- flip on more frequent control bit
        5046, # cnot(not(2),3) -- flip on less frequent control bit
        11536, # weakly systematic -- if positional=True, no different from strongly systematic
        10000, # nonsystematic
        15000,
        20000,
        25000,
        30000, # nonsystematic
    ]
    df, support, probs = learn_from_samples(K=3, V=2, num_samples=num_samples, targets=targets*num_runs, **kwds)
    grammars = list(itertools.permutations(support))
    def gen():
        for target in targets:
            yield {
                'target': target,
                'grammar': grammars[target],
                'ee': il.ee(il.curves_from_sequences(grammars[target], probs)),
            }
    return df, pd.DataFrame(gen())

def learn_from_samples(
        targets=[0],
        K=3,
        V=2,
        T=1,
        split_alpha=1,
        maxlen=2,
        num_samples=10000,
        redundancy=1,
        positional=True):

    # This is cryptography. The grammar is the key to be deciphered.
    # Given uniformly distribution s = f_K(m), and knowledge of p(m),
    # and lots of samples of s, how long to determine K?

    # Intuitively if the data distribution is flat, then the key is harder to recover,
    # because there are more keys that give rise to that distribution: structure-preserving
    # keys are rare. 
    alphabet = [chr(65+v) for v in range(V)]
    if K > 2:
        probs = 1/T * np.random.randn(V)
    else:
        probs = scipy.special.log_softmax([-1/T, 1/T])
    character_source = pmonad.Enumeration(list(zip(alphabet, probs)))
    string_source = pmonad.Enumeration.ret("")
    
    @string_source.lift_ret
    def concatenate_distinctly(c, x):
        if positional:
            return "".join([c, chr(len(c)*V + ord(x))*redundancy])
        else:
            return "".join([c, x*redundancy])
        
    for k in range(K):
        string_source = string_source >> (lambda c:
            character_source >> (lambda x:
            concatenate_distinctly(c, x)))

    source_probs = [p for _, p in string_source]
    strings = [s for s, _ in string_source]    
    source = pmonad.Enumeration(list(enumerate(source_probs)))
    grammars_support = list(itertools.permutations(strings))
    
    @source.lift_ret
    def produce(g, m):
        return grammars_support[g][m]

    context_size = source.exponential(range(maxlen), alpha=split_alpha)
    
    def likelihood(g):
        utterances = source >> (lambda m: produce(g, m))
        # adjust probabilities
        adjusted = type(source)(
            (x, source.field.mul(source.field.from_p(len(x)+1), p))
            for x, p in utterances.values
        )
        return adjusted >> (lambda utt:
            context_size >> (lambda k:
            source.uniform(pmonad.increments(utt + '#', k))))

    # arrange into joint probabilities of shape GO, evidence of shape O
    print("Calculating likelihood array...", file=sys.stderr)
    likelihood_array = np.array([
        [source.field.to_log(p) for _, p in sorted(likelihood(g))]
        for g in tqdm.tqdm(range(len(grammars_support)))
    ])
    likelihood_index = sorted([v for v, _ in sorted(likelihood(0))])

    def gen():
        for target in rfutils.interruptible(targets):
            print("Running target grammar %d" % target, file=sys.stderr)
            target_grammar = grammars_support[target]
            prior_array = -log(len(grammars_support)) * np.ones(len(grammars_support))
            for i in tqdm.tqdm(range(num_samples)):
                observed = np.random.choice(
                    range(len(likelihood_index)),
                    p=np.exp(likelihood_array[target])
                )
                joint = prior_array + likelihood_array[:, observed] # shape G
                posterior_array = scipy.special.log_softmax(joint, -1) # shape G
                yield {
                    'target': str(target),
                    'i': i,
                    'data': likelihood_index[observed],
                    'posterior': posterior_array[target],
                    'entropy': -(np.exp(posterior_array) * posterior_array).sum(),
                }
                prior_array = posterior_array
    return pd.DataFrame(list(gen())), strings, source_probs
        
def tap(x):
    print(x)
    return x
    
def head_direction_consistency(
        num_cats=5,
        num_heads=5,
        max_modifiers=1,
        with_brackets=True,
        T=1,
        bound=2,
        with_delimiter=True):
    # show: by-category consistent > consistent
    # comparing to total inconsistency is hard because it increases the entropy rate

    # need to try with arity-3 trees so that this doesn't just reduce to the regular vs. CFG comparison...
    def rules(right_branch):
        for phrase in range(num_cats):
            phrase_label = f"{phrase}P"
            for head in range(num_heads):
                yield pcfg.Rule(phrase_label, ["(", f"h{phrase}_{head}", ")"])
                for phrase2 in range(num_cats):
                    if right_branch[phrase2]:
                        yield pcfg.Rule(phrase_label, ["(", f"h{phrase}_{head}", f"{phrase2}P", ")"])
                    else:
                        yield pcfg.Rule(phrase_label, ["(", f"{phrase2}P", f"h{phrase}_{head}", ")"])

    mask = [np.random.random() < .5 for _ in range(num_cats)]
    inconsistent_rules = list(rules(mask))
    consistent_rules = list(rules([True]*num_cats))
    probs = np.exp(np.random.randn(len(consistent_rules)))
    inconsistent_grammar = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(inconsistent_rules, probs), bound, start="0P")
    consistent_grammar = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(consistent_rules, probs), bound, start="0P")
    inconsistent_forms = pcfg_forms(inconsistent_grammar, with_delimiter=with_delimiter)
    consistent_forms = pcfg_forms(consistent_grammar, with_delimiter=with_delimiter)
    return inconsistent_forms, consistent_forms, mask
                
def dutch_vs_german(
        num_n=5,
        num_v=5,
        p_embed=.5,
        independent=False,
        T=1,
        bound=3,
        with_delimiter=True):
    """ Nested vs. cross-serial dependencies, as in German vs. Dutch """
    # When independent=True, Dutch and German come out exactly the same---same dependency length effect.
    # When independent=False, Dutch is slightly better than German.
    def rules():
        for n, v in itertools.product(range(num_n), range(num_v)):
            yield pcfg.Rule("S", [f"N{n}", f"V{v}"])
            yield pcfg.Rule("S", [f"N{n}", "S", f"V{v}"])
    the_rules = list(rules())
    if not independent:
        probs = (np.exp(1/T*np.random.randn(len(the_rules) // 2, 2)) * np.array([1-p_embed, p_embed])).flatten()
    else:
        probs = (np.exp(1/T*np.random.randn(len(the_rules) // 2)[:, None]) * np.array([1-p_embed, p_embed])).flatten()
    german_grammar = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(the_rules, probs), bound, start="S")
    forms = pcfg_forms(german_grammar, with_delimiter=with_delimiter)
    forms.columns = ['german', 'p']
    
    def convert_to_dutch(form):
        form = form.strip('#')
        midpoint = len(form)//2
        raw_form = form[:midpoint] + form[midpoint:][::-1]
        if with_delimiter == 'left':
            return '#' + raw_form
        elif with_delimiter:
            return '#' + raw_form + '#'
        else:
            return raw_form

    ds = sh.DeterministicScramble()
    
    forms['dutch'] = forms['german'].map(convert_to_dutch)
    forms['scramble'] = forms['german'].map(ds.shuffle)
    return forms

def random_xbar_pcfg(num_cats, num_heads, num_adjuncts, adjunct_optional=False, T=1, bound=1):
    """ Xbar-style random PCFG, with a matched finite-state grammar.
    If the PCFG is unambiguous, then its entropy is the same as its finite-state equivalent.
    """
    # XP -> y X'
    # X' -> (ZP) x
    # where each XP is associated with disjoint sets of adjuncts y and heads x.
    def rules():
        for category in range(num_cats):
            bar_label = f"{category}'"
            for head in range(num_heads):
                head_label = f"{category}_{head}"
                yield pcfg.Rule(bar_label, [head_label]) # X' -> x                
                for subcategory in range(num_cats):
                    yield pcfg.Rule(bar_label, [f"{subcategory}P", head_label]) # X' -> ZP x
                    
            phrase_label = f"{category}P"
            if adjunct_optional:
                yield pcfg.Rule(phrase_label, [bar_label]) # XP -> X'
            for adjunct in range(num_heads):
                adjunct_label = f"a{category}_{adjunct}"
                yield pcfg.Rule(phrase_label, [adjunct_label, bar_label]) # XP -> y X'
                
    the_rules = list(rules())
    reg_rules = [ # reverse if it's a bar level rule, yielding a regular grammar
        pcfg.Rule(rule.lhs, list(reversed(rule.rhs))) if "'" in rule.lhs else rule # X' -> x ZP
        for rule in the_rules 
    ]
    probs = np.exp(np.random.randn(len(the_rules)))

    if bound is None:
        cf = pcfg.make_pcfg(pcfg.PSpaceEnumeration, zip(the_rules, probs), start="0P")
        reg = pcfg.make_pcfg(pcfg.PSpaceEnumeration, zip(reg_rules, probs), start="0P")        
    else:
        cf = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(the_rules, probs), bound, start="0P")
        reg = pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, zip(reg_rules, probs), bound, start="0P")
        
    return cf, reg

def pcfg_forms(pcfg, with_delimiter=True):
    def recode(x, _seen={}):
        if x in _seen:
            return _seen[x]
        else:
            _seen[x] = len(_seen)
            return _seen[x]
    def process_form(xs):
        if with_delimiter == 'left':
            return '#' + "".join(chr(65+recode(x)) for x in xs)
        elif with_delimiter:
            return '#' + "".join(chr(65+recode(x)) for x in xs) + '#'
        else:
            return "".join(chr(65+recode(x)) for x in xs)
    forms, probs = zip(*pcfg.distribution())
    forms = map(process_form, forms)
    return pd.DataFrame({'form': forms, 'p': probs})

def analyze_pcfg(pcfg, with_delimiter=True):
    df = pcfg_forms(pcfg)
    curves = il.curves_from_sequences(df['form'], df['p'])
    return curves

def random_cnf_pcfg(num_nt, num_t, T=1, bound=None):
    # for each nt, a distribution over nts and ts

    # NT -> NT NT, or
    # NT -> T.
    nt_grammar = np.exp(1/T*np.random.randn(num_nt, num_nt, num_nt))
    t_grammar = np.exp(1/T*np.random.randn(num_nt, num_t))
    ts = range(num_t)
    nts = range(num_nt)
    rules = [
        (pcfg.Rule(str(num_t + i), (str(j),)), t_grammar[i, j])
        for i, j in itertools.product(nts, ts)
    ]
    rules.extend(
        (pcfg.Rule(str(num_t + i), (str(num_t + j), str(num_t + k))), nt_grammar[i,j,k])
        for i, j, k in itertools.product(nts, nts, nts)
    )

    if bound is None:
        return pcfg.make_pcfg(pcfg.PSpaceEnumeration, rules, start=str(num_t))
    else:
        return pcfg.make_bounded_pcfg(pcfg.PSpaceEnumeration, rules, bound, start=str(num_t))
    
    
    
    
    

