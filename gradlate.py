#!/usr/bin/env python3
import sys
import getopt
from collections import Counter as Counter
import re
from nltk.stem.snowball import SnowballStemmer as SS

word_separator = re.compile('[ \n\'".,/:;!?()]+')

def help_exit():
    print('test.py -f <language>:<from_translation> -t <language>:<to_translation> -o <outputfile>')
    sys.exit(2)

def normalize(w): return w.lower()

def count_freqs(text, stemmer):
    ctr = Counter([stemmer.stem(w) for w in word_separator.split(text)])
    return [t for t in ctr.most_common() if t[1] > 5 and len(t[0]) > 1]

def gradlate(ft, fs, tt, ts, out):
    fs.maxCacheSize = 1000000
    ts.maxCacheSize = 1000000
    f_freqs = count_freqs(ft, fs)
    t_freqs = count_freqs(tt, ts)
    print(f_freqs[0:200])
    print(t_freqs[0:200])

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

    with open(from_path, 'r') as from_file:
        with open(to_path, 'r') as to_file:
            if out_path:
                with open(out_path, 'w') as out_file:
                    gradlate(from_file.read(), SS(from_lang), to_file.read(), SS(to_lang), out_file)
            else:
                gradlate(from_file.read(), SS(from_lang), to_file.read(), SS(to_lang), sys.stdout)

