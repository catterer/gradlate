#!/usr/bin/env python3
import sys
import getopt
from collections import Counter as Counter
import re
from nltk.stem.snowball import SnowballStemmer as SS
from nltk.translate.gale_church import align_blocks,align_texts

word_separator = re.compile('[ \n\'".,/:;!?()]+')
sentence_separator = re.compile('[".!?()]+')
block_separator = re.compile('\n\n\n\n')
empty_sentence = re.compile('^\s*$')

def help_exit():
    print('test.py -f <language>:<from_translation> -t <language>:<to_translation> -o <outputfile>')
    sys.exit(2)

def normalize(w): return w.lower()

class Block:
    def __init__(self, block_raw):
        self.raw = block_raw
        self.sentences = [s for s in sentence_separator.split(self.raw) if not empty_sentence.match(s)]
        self.sentence_lengths = [len(s) for s in self.sentences]
        self.words = word_separator.split(self.raw)

class Text:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.raw_text = f.read()
        self.blocks = [Block(b) for b in block_separator.split(self.raw_text)]

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

    print(align_blocks(ft.blocks[2].sentence_lengths, tt.blocks[2].sentence_lengths))

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

    if out_path:
        with open(out_path, 'w') as out_file:
            gradlate(Text(from_path), Text(to_path), out_file)
    else:
        gradlate(Text(from_path), Text(to_path), sys.stdout)

