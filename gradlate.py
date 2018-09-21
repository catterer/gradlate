#!/usr/bin/env python3
import sys
import pickle
import getopt
from collections import Counter as Counter
import re
from nltk.stem.snowball import SnowballStemmer as SS
from nltk.translate.gale_church import align_blocks,align_texts
from nltk.translate.ibm3 import IBMModel3
from nltk.translate.ibm2 import IBMModel2
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.api import AlignedSent

word_separator = re.compile('[\s\n\'“"–.,/:;!?()]+')
sentence_separator = re.compile('[“".!?()]+')
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

class WordXn:
    def __init__(self, f, t):
        self.f = f
        self.t = t
    def __repr__(self):
        return '{} = {}'.format(self.f, self.t)

def select_good_translations(model):
    res = []
    for from_w in model.translation_table.keys():
        for to_w in model.translation_table[from_w].keys():
            if model.translation_table[from_w][to_w] > 0.9:
                res.append(WordXn(from_w, to_w))

    f_count = Counter([t.f for t in res])
    t_count = Counter([t.t for t in res])
    
    return [t for t in res if f_count[t.f] == 1 and t_count[t.t] == 1]


class TextXn:
    def __init__(self, text_f, text_t):
        if len(text_f.blocks) != len(text_t.blocks):
            raise Exception('different amount of blocks in texts')

        self.text_f = text_f
        self.text_t = text_t
        self.aligned_blocks = {}
        self.model = None
        self.bitex = []

    def blocks_number(self):
        return len(self.text_f.blocks)

    def align_block(self, block_n):
        fb = self.text_f.blocks[block_n]
        tb = self.text_t.blocks[block_n]
        self.aligned_blocks[block_n] = align_blocks(fb.stnc_lengths_char, tb.stnc_lengths_char)

    def build_bitex(self):
        self.bitex = []
        for b_id in self.aligned_blocks.keys():
            flast = None
            tlast = None

            for p in self.aligned_blocks[b_id]:
                if p[0] == flast or p[1] == tlast:
                    continue
                (flast,tlast) = (p[0], p[1])
                fb = self.text_f.blocks[b_id]
                tb = self.text_t.blocks[b_id]
                self.bitex.append(AlignedSent(fb.sentences[p[0]].words, tb.sentences[p[1]].words))

    def train_model(self):
        self.model = IBMModel2(self.bitex, 5)
        for t in select_good_translations(self.model):
            print(t)

    def dump(self, fname):
        with open(fname, 'wb') as fd:
            pickle.dump(self, fd)


if __name__ == '__main__':
    try:
        opts, _ = getopt.getopt(sys.argv[1:],'hf:t:o:x:',['from=','to=','out=','xn='])
    except getopt.GetoptError:
        help_exit()

    from_path = None
    from_lang = None
    to_path = None
    to_lang = None
    out_path = None
    xn_path = None

    for k,v in opts:
        if k == '-h': help_exit()
        if k == '-f': (from_lang, from_path) = v.split(':')
        if k == '-t': (to_lang, to_path) = v.split(':')
        if k == '-o': out_path = v
        if k == '-x': xn_path = v

    if not xn_path and not all([from_path, to_path]):
        help_exit()

    if xn_path:
        with open(xn_path, 'rb') as xn_file:
            xn = pickle.load(xn_file)
    else:
        from_text = Text(from_lang, from_path)
        to_text = Text(to_lang, to_path)
        xn = TextXn(from_text, to_text)

#       for i in range(0, xn.blocks_number()):
#           xn.align_block(i)
#           xn.dump('.trash/aligned.{}'.format(i))

    xn.build_bitex()
    xn.dump('.trash/bitex')

    xn.train_model()
    xn.dump('.trash/model')

