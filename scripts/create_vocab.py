# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')

import argparse
import os 
import json
from coreLib.languages import languages
from coreLib.utils import LOG_INFO
#-----------------------------------------------------------------------------------
def main(args):
    language=args.language
    lang=languages[language]
    vocab_json  =f"../{language}.json"
    gvocab=lang.components
    cvocab=lang.unicodes

    LOG_INFO(f"Unicode:{len(cvocab)}")
    LOG_INFO(f"Grapheme:{len(gvocab)}")

    # config 
    config={'gvocab':gvocab,'cvocab':cvocab}
    with open(vocab_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)
#-----------------------------------------------------------------------------------
if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Creating vocab.json for language")
    parser.add_argument("language", help="the specific language to use")
    args = parser.parse_args()
    main(args)