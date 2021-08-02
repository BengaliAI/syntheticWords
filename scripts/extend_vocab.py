#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')
import json
import string
from coreLib.utils import LOG_INFO
#--------------------
# globals
#--------------------
vocab_json  ="../vocab.json"

eng=list(string.ascii_lowercase)
eng+=[str(i) for i in range(10)]

with open(vocab_json) as f:
    curr_vocab = json.load(f)["gvocab"]

vocab=list(set(curr_vocab[1:]+eng))
gvocab=sorted(vocab)
# unicodes
cvocab=[]
for g in gvocab:
    for i in g:
        if i not in cvocab:
            cvocab.append(i)

cvocab=sorted(cvocab)
# pad both side
gvocab=['']+gvocab
cvocab=['']+cvocab


LOG_INFO(f"Unicode:{len(cvocab)}")
LOG_INFO(f"Grapheme:{len(gvocab)}")

# config 
config={'gvocab':gvocab,'cvocab':cvocab}
with open(vocab_json, 'w') as fp:
    json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)