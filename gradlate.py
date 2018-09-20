#!/usr/bin/env python3
import sys
import getopt
from collections import Counter as Counter
import re

word_separator = re.compile('[ \n\'".,/:;!?()]+')

def help_exit():
    print('test.py -f <from_translation> -t <to_translation> -o <outputfile>')
    sys.exit(2)

def normalize(w): return w.lower()

def count_freqs(text):
    ctr = Counter([normalize(w) for w in word_separator.split(text)])
    return ctr.most_common()

def gradlate(ft, tt, out):
    f_freqs = count_freqs(ft)
    t_freqs = count_freqs(tt)
    print(f_freqs[0:200])
    print(t_freqs[0:200])

if __name__ == '__main__':
    try:
        opts, _ = getopt.getopt(sys.argv[1:],'hf:t:o:',['from=','to=','out='])
    except getopt.GetoptError:
        help_exit()

    from_path = None
    to_path = None
    out_path = None

    for k,v in opts:
        if k == '-h': help_exit()
        if k == '-f': from_path = v
        if k == '-t': to_path = v
        if k == '-o': out_path = v

    if not all([from_path, to_path]):
        help_exit()

    with open(from_path, 'r') as from_file:
        with open(to_path, 'r') as to_file:
            if out_path:
                with open(out_path, 'w') as out_file:
                    gradlate(from_file.read(), to_file.read(), out_file)
            else:
                gradlate(from_file.read(), to_file.read(), sys.stdout)

