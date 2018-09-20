#!/usr/bin/env python3
import sys
import getopt
from collections import Counter as Counter
import re
from nltk.stem.snowball import SnowballStemmer as SS
from nltk.translate.gale_church import align_blocks,align_texts
from nltk.translate.ibm3 import IBMModel3
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.api import AlignedSent

word_separator = re.compile('[ \n\'".,/:;!?()]+')
sentence_separator = re.compile('[".!?()]+')
block_separator = re.compile('\n\n\n\n')

def help_exit():
    print('test.py -f <language>:<from_translation> -t <language>:<to_translation> -o <outputfile>')
    sys.exit(2)

def normalize(w): return w.lower()

class Sentence:
    def __init__(self, stem, stc_raw):
        self.raw = stc_raw
        self.words = [stem.stem(w) for w in word_separator.split(self.raw)]

class Block:
    def __init__(self, stem, block_raw):
        self.raw = block_raw
        self.sentences = [Sentence(stem, s) for s in sentence_separator.split(self.raw)]
        self.stnc_lengths_char = [len(s.raw) for s in self.sentences]

class Text:
    def __init__(self, lang, filename):
        self.lang = lang
        self.stem = SS(lang)
        with open(filename, 'r') as f:
            self.raw_text = f.read()
        self.blocks = [Block(self.stem, b) for b in block_separator.split(self.raw_text)]

def chunk_list(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def gradlate(ft, tt, out):
    if len(ft.blocks) != len(tt.blocks):
        raise Exception('different amount of blocks in texts')

    block_n = 2
    fb = ft.blocks[block_n]
    tb = tt.blocks[block_n]

    pairs = align_blocks(fb.stnc_lengths_char, tb.stnc_lengths_char)
#   pairs = aligned_pairs[round(len(aligned_pairs)/2) + 1]
#   print(fb.sentences[pairs[0]])
#   print(tb.sentences[pairs[1]])

    flast = None
    tlast = None

    bitex = []
    for p in pairs:
        if p[0] == flast or p[1] == tlast:
            continue
        (flast,tlast) = (p[0], p[1])
        bitex.append(AlignedSent(fb.sentences[p[0]].words, tb.sentences[p[1]].words))

#   for p in bitex:
#       print(p)

    ibm = IBMModel1(bitex, 5)

    for from_w in ibm.translation_table.keys():
        for to_w in ibm.translation_table[from_w].keys():
            p = ibm.translation_table[from_w][to_w]
            if p > 0.5:
                print('{} = {} ({})'.format(from_w, to_w, p))

if __name__ == '__main__':
    try:
        opts, _ = getopt.getopt(sys.argv[1:],'hf:t:o:',['from=','to=','out='])
    except getopt.GetoptError:
        help_exit()

    from_path = None
    from_lang = None
    to_path = None
    to_lang = None
    out_path = None

    for k,v in opts:
        if k == '-h': help_exit()
        if k == '-f': (from_lang, from_path) = v.split(':')
        if k == '-t': (to_lang, to_path) = v.split(':')
        if k == '-o': out_path = v

    if not all([from_path, to_path]):
        help_exit()

    from_text = Text(from_lang, from_path)
    to_text = Text(to_lang, to_path)

    out_file = sys.stdout
    if out_path:
        out_file = open(out_path, 'w')

    gradlate(from_text, to_text, out_file)
    out_file.close()

