import sys
import re
import difflib

FILENAME = "anipa_to_ipa.tsv"

def read_data(filename):
    with open(filename, 'rt') as infile:
        parts = (line.strip().split() for line in infile)
        d = {code : phoneme for code, phoneme in parts}
    return d

CHART = read_data(FILENAME)
WORD_BOUNDARY = "#"
MORPHEME_BOUNDARY = "%"
PHONEME_BOUNDARY = "_"

LONG = "ː"        
UNVOICE = str("\u0325")
VOICE = "̬"
STRESS = "1"
SECONDARY_STRESS = "2"
DENTAL = str("\u032a")
EJECTIVE = "ʼ"
BREATHY_CONSONANT = "ʱ"
BREAHTY_VOWEL = "◌̤"
PHARYNGEALIZED = "ˤ"
LABIALIZED = "ʷ"
PALATIZED = "ʲ"
VELARIZED = "ˠ"
ASPIRATED = "ʰ"
LATERAL = "ˡ"
FALLING_TONE = "˥˩"
HIGH_TONE = "˥"
MID_TONE = "˧"
LOW_TONE = "˩"
RISING_TONE = "˩˥"
NASALIZED = str("\u0303")

def segment(word):
    if word.startswith(WORD_BOUNDARY) or word.startswith(MORPHEME_BOUNDARY):
        initial, *word = word
        word = "".join(word)
    else:
        initial = ""
    
    if word.endswith(WORD_BOUNDARY) or word.endswith(MORPHEME_BOUNDARY):
        *word, final = word
        word = "".join(word)
    else:
        final = ""

    yield initial
    morphemes = re.split("[%s%s]" % (MORPHEME_BOUNDARY, WORD_BOUNDARY), word)
    first = True
    for morpheme in morphemes:
        if not first:
            yield MORPHEME_BOUNDARY # TODO keep track of the right kind of boundary
        yield from morpheme.split(PHONEME_BOUNDARY)
        first = False
    yield final

def unsegment(codes):
    return "_".join(codes).replace("_#", "#").replace("#_", "#").replace("%_", "%").replace("_%", "%")

def convert_file(*filenames):
    if filenames:
        files = map(open, filenames)
    else:
        files = [sys.stdin]
    for file in files:
        for word in file:
            yield convert_word(word.strip())
    
def convert_word(word):
    return "".join(map(convert_code, segment(word)))

def convert_code(code):    
    differ = difflib.Differ()
    if not code:
        return ""
    elif code in CHART: 
        return CHART[code]
    elif code == WORD_BOUNDARY:
        return WORD_BOUNDARY
    elif code == MORPHEME_BOUNDARY:
        return MORPHEME_BOUNDARY
    else:
        # find closest thing in data, then add appropriate diacritics
        closest_codes = difflib.get_close_matches(code, CHART.keys(), 5)
        for closest_code in closest_codes:
            closest_ipa = CHART[closest_code]
            diff = set(chunk for chunk in differ.compare(closest_code, code) if not chunk.startswith(" "))
            if '- v' in diff and '+ f' in diff:
                closest_ipa += UNVOICE
                diff.remove('- v')
                diff.remove('+ f')
            if '- f' in diff and '+ v' in diff:
                closest_ipa += VOICE
                diff.remove('- v')
                diff.remove('+ f')                
            if '- o' in diff and '+ n' in diff:
                closest_ipa += NASALIZED
                diff.remove('- o')
                diff.remove('+ n')                
            if closest_code.startswith('V') and '- N' in diff and '+ P' in diff:
                closest_ipa += STRESS
                diff.remove('- N')
                diff.remove('+ P')
            if closest_code.startswith('V') and '- N' in diff and '+ S' in diff:
                closest_ipa += SECONDARY_STRESS
                diff.remove('- N')
                diff.remove('+ S')                
            if closest_code.startswith('C') and '- T' in diff and '+ D' in diff:
                closest_ipa += DENTAL
                diff.remove('- T')
                diff.remove('+ D')
            if closest_code.startswith("C") and "+ A" in diff and '- X' in diff:
                closest_ipa += PHARYNGEALIZED
                diff.remove('+ A')
                diff.remove('- X')
            if closest_code.startswith("C") and "+ W" in diff and '- X' in diff:
                closest_ipa += LABIALIZED
                diff.remove('+ W')
                diff.remove('- X')
            if closest_code.startswith("C") and "+ L" in diff and "- N" in diff:
                closest_ipa += LATERAL
                diff.remove("+ L")
                diff.remove("- N")
            if closest_code.startswith("C") and "+ Y" in diff and '- X' in diff:
                closest_ipa += PALATIZED
                diff.remove('+ Y')
                diff.remove('- X')
            if closest_code.startswith("C") and "+ G" in diff and '- X' in diff:
                closest_ipa += VELARIZED
                diff.remove('+ G')
                diff.remove('- X')
            if closest_code.startswith("C") and "+ j" in diff and '- f' in diff:
                closest_ipa += EJECTIVE
                diff.remove('+ j')
                diff.remove('- f')
            if closest_code.startswith("C") and "+ j" in diff and '- v' in diff:
                closest_ipa += EJECTIVE
                diff.remove('+ j')
                diff.remove('- v')
            if closest_code.startswith("C") and "+ b" in diff and '- v' in diff:
                closest_ipa += BREATHY_CONSONANT
                diff.remove('+ b')
                diff.remove('- v')
            if closest_code.startswith("V") and "+ b" in diff and '- v' in diff:
                closest_ipa += BREATHY_VOWEL
                diff.remove('+ b')
                diff.remove('- v')                                
            if closest_code.startswith("C") and "+ h" in diff and '- f' in diff:
                closest_ipa += ASPIRATED
                diff.remove('+ h')
                diff.remove('- f')
            if closest_code.startswith("C") and "+ h" in diff and '- v' in diff:
                closest_ipa += ASPIRATED
                diff.remove('+ h')
                diff.remove('- v')                 
            if closest_code.startswith("V") and "+ H" in diff and '- N' in diff:
                closest_ipa += HIGH_TONE
                diff.remove("+ H")
                diff.remove("- N")
            if closest_code.startswith("V") and "+ M" in diff and "- N" in diff:
                closest_ipa += MID_TONE
                diff.remove("+ M")
                diff.remove("- N")
            if closest_code.startswith("V") and "+ L" in diff and "- N" in diff:
                closest_ipa += LOW_TONE
                diff.remove("+ L")
                diff.remove("- N")
            if closest_code.startswith("V") and "+ R" in diff and "- N" in diff:
                closest_ipa += RISING_TONE
                diff.remove("+ R")
                diff.remove("- N")
            if closest_code.startswith("V") and "+ F" in diff and "- N" in diff:
                closest_ipa += FALLING_TONE
                diff.remove("+ F")
                diff.remove("- N")                                                                
            if '- 1' in diff and '+ 2' in diff:
                closest_ipa += LONG
                diff.remove('- 1')
                diff.remove('+ 2')
            if '- 1' in diff and '+ 3' in diff:
                closest_ipa += LONG + LONG
                diff.remove('- 1')
                diff.remove('+ 3')                
            if not diff:
                return closest_ipa 

        raise ValueError("Could not convert to IPA: %s" % code)

if __name__ == '__main__':
    for line in convert_file(*sys.argv[1:]):
        print(line)

