import sys
import itertools
from collections import Counter

import tqdm
import numpy as np
import pandas as pd

import infoloc as il
import shuffles as sh

# Need langs with
# (1) A ground truth source of morphology.
# (2) A large UD corpus with morphological features annotated
# (3) Morphological features Case, Number, and Psor. 

# Currently:
# Finnish
# Hungarian
# Turkish
# Latin

# Candidates:
# Ancient Greek -- complex, articles, ...
# Arabic -- need to transliterate, what about gender?
# Icelandic -- complex...

# No-go:
# Amharic -- no feats on nouns in UD
# Chukchi -- no feats
#

STEM = "X"

def freeze(d):
    return frozenset(d.items())

def read_features(
        filename,
        required_pos={'NOUN'},
        required_labels={'Case', 'Number'},
        include_labels={'Case', 'Number', 'Number[psor]', 'Person[psor]'},
        reject_labels={'PronType'}):
    def gen(lines):
        for line in lines:
            if line[0].isdigit(): # it's a token
                index, form, lemma, pos, pos2, features, head, rel, *_ = line.split("\t")
                if pos in required_pos and '=' in features:
                    d = dict(f.split("=") for f in features.split("|"))
                    d['pos'] = pos
                    if all(req in d for req in required_labels) and not any(bad in d for bad in reject_labels):
                        yield {label:value for label, value in d.items() if label in include_labels or label == 'pos'}
    with open(filename) as infile:
        result = Counter(map(freeze, gen(infile)))
    return result

def convert_ud_um(d):
    # CURRENTLY ONLY NOUNS!
    conversion = {
        'Sing' : 'SG',
        'Plur' : 'PL',
        'NOUN' : 'N',
        
        'VERB' : 'V',
        'Pres': 'PRS',
        'Past': 'PST',
        #'Sub': 'SBJV',
        'Cnd': 'COND',
        
        'Ins': 'INST',  # val
        'Cau': 'PRP', # ért
        'Ter': 'TERM', # ig
        'Par': 'PRT', 
        'Tra' : 'TRANS', # vá        
        'Ess': 'FRML',  # essive in Finnish
        'Abe': 'PRIV',  # abessive in Finnish

        'Abs': 'FRML', # ként
        'Dis': 'FRML', # ként

        'Sub': 'ON+ALL', # ra
        'Del': 'ON+ABL', # ról        
        'Sup': 'ON+ESS', # on
        
        'All': 'AT+ALL',  # allative, hoz   
        'Abl': 'AT+ABL',  # ablative, tól
        'Ade': 'AT+ESS',  # adessive, nál
        
        'Ill': 'IN+ALL',  # illative, ba
        'Ela': 'IN+ABL', # elative, ból  
        'Ine': 'IN+ESS', # inessive, ben
        
    }
    def convert(x):
        if x in conversion:
            return conversion[x].upper()
        else:
            return x.upper()
    def gen():
        if d['pos'] == 'NOUN':
            yield convert(d['pos'])
            yield convert(d['Case'])
            yield convert(d['Number'])
            if 'Number[psor]' in d and 'Person[psor]' in d:
                yield 'PSS%d%s' % (int(d['Person[psor]']), 'P' if d['Number[psor]'] == 'Plur' else 'S')
        elif d['pos'] == 'VERB':
            yield convert(d['Tense'])
            yield convert(d['Mood'])
            yield convert(d['Person'])
            yield convert(d['Number'])
                
    return ";".join(gen())

def desired_bundles(bundles):
    """ Logical product of all attested features. """
    # TODO: could cause trouble with very spotty features---the optional one needs to be the final one!
    the_product = itertools.product(*map(set, itertools.zip_longest(*[s.split(";") for s in bundles])))
    return {";".join(filter(None, bundle)) for bundle in the_product}

def expand_stem(form, k):
    return form.replace(STEM, STEM*k)

def experiment(
        ud_filename,
        um_filename,
        alpha=1/2,
        num_samples=50,
        len_granularity=1,
        stem_redundancy=1,
        include_psor=True):
    if include_psor:
        counts = read_features(ud_filename, include_labels={'Case', 'Number', 'Number[psor]', 'Person[psor]'})
    else:
        counts = read_features(ud_filename, include_labels={'Case', 'Number'})
    counts_um = Counter({convert_ud_um(dict(bundle)):count for bundle, count in counts.items()})
    desired = desired_bundles(counts_um)
    forms = pd.read_csv(um_filename, sep="\t")
    forms['form'] = forms['form'].map(lambda s: expand_stem(s, stem_redundancy))
    forms = forms[forms['features'].map(lambda f: f in desired)]
    forms['count'] = forms['features'].map(lambda f: counts_um[f] + alpha)
    forms['len'] = forms['form'].map(len)
    forms = forms.sort_values('len', ignore_index=True) # WTF
    forms.to_csv(sys.stderr)
    def conditions():
        yield 'real', 0, il.curves_from_sequences(forms['form'], forms['count'])
        for i in tqdm.tqdm(range(num_samples)):
            yield 'dscramble', i, il.curves_from_sequences(forms['form'].map(sh.DeterministicScramble(i).shuffle), forms['count'])
            nonsys_forms = pd.Series(np.random.permutation(forms['form'].values))
            forms['forms_nonsys'] = nonsys_forms
            yield 'nonsys', i, il.curves_from_sequences(nonsys_forms, forms['count'])
            new_forms = []
            lenclass = forms['len'] // len_granularity
            for length in lenclass.drop_duplicates(): # ascending order
                mask = lenclass == length
                shuffled_forms = np.random.permutation(forms[mask]['form'])
                new_forms.extend(shuffled_forms)
            new_forms = pd.Series(new_forms)
            forms['forms_nonsysl'] = new_forms 
            yield 'nonsysl', i, il.curves_from_sequences(new_forms, forms['count'])
            
    def gen():
        for name, i, df in conditions():
            df['type'] = name
            df['sample'] = i
            yield df
    return forms, pd.concat(list(gen()))

# Turkish UM has schema N;CASE;NUMBER;PERSON
# where CASE = {NOM, ACC, GEN, ABL, LOC, DAT}
# NUMBER = {SG=Sing, PL=Plur}
# PERSON = {PSS1S, PSS2S, PSS3S, PSS1P, PSS2P, PSS3P}

def run(num_samples=15000):
    finnish_forms, finnish_curves = experiment("/Users/canjo/data/cliqs/ud-treebanks-v2.8/fi/all.conllu", "data/fin", num_samples=num_samples) # talo
    finnish_curves['lang'] = 'Finnish'
    
    turkish_forms, turkish_curves = experiment("/Users/canjo/data/cliqs/ud-treebanks-v2.12/UD_Turkish-Penn/all.conllu", "data/tur", num_samples=num_samples) # av
    turkish_curves['lang'] = 'Turkish'
    
    latin_forms, latin_curves = experiment("/Users/canjo/data/cliqs/ud-treebanks-v2.8/la/all.conllu", "data/lat", num_samples=num_samples) # puella
    latin_curves['lang'] = 'Latin'
    
    hun_forms, hun_curves = experiment("/Users/canjo/data/cliqs/ud-treebanks-v2.8/hu/all.conllu", "data/hun", num_samples=num_samples) # cél
    hun_curves['lang'] = 'Hungarian'
    return pd.concat([turkish_curves, finnish_curves, latin_curves, hun_curves])

if __name__ == '__main__':
    run().to_csv(sys.stdout)

# NOUNS

                    

            
# Finnish finite verbs have features (active finite positive only):
# tense; mood; person; number

# features to exclude: PASS; NEG; 
# Active/Passive (active unmarked in UD; passive is maybe not finite?)
# Polarity (periphrastic--in UM but not in UD)

