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

import argparse
import os 
import json
import pandas as pd 

from tqdm import tqdm
from ast import literal_eval

from coreLib.utils import LOG_INFO
from coreLib.dataset import DataSet
from coreLib.synthetic import createWords
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    pad_height  =   int(args.pad_height)
    num_samples =   int(args.num_samples)
    # dataset object
    ds=DataSet(data_dir=data_path)
    LOG_INFO("Creating bangla graphemes data")
    createWords(iden="bangla.graphemes",
                df=ds.bangla.graphemes.df,
                img_dir=ds.bangla.graphemes.dir,
                save_dir=save_path,
                font_path=ds.bangla.font,
                img_dim=(img_height,img_width),
                comp_dim=img_height,
                pad_height=pad_height,
                top_exts=ds.bangla.top_exts,
                bot_exts=ds.bangla.bot_exts,
                dictionary=ds.bangla.dictionary,
                num_samples=num_samples)
    LOG_INFO("Creating bangla numbers data")
    createWords(iden="bangla.numbers",
                df=ds.bangla.numbers.df,
                img_dir=ds.bangla.numbers.dir,
                save_dir=save_path,
                font_path=ds.bangla.font,
                img_dim=(img_height,img_width),
                comp_dim=img_height,
                pad_height=pad_height,
                top_exts=ds.bangla.top_exts,
                bot_exts=ds.bangla.bot_exts,
                dictionary=None,
                valid_graphemes=ds.bangla.number_values,
                num_samples=num_samples)            

    # config 
    config={'gvocab':ds.bangla.gvocab,}
    with open("../vocab.json", 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)

    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training: Bangla Synthetic (numbers and graphemes) Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the source data folder ")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--pad_height",required=False,default=20,help ="pad height for each grapheme for alignment correction: default=20")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension of word images: default=512")
    parser.add_argument("--num_samples",required=False,default=100000,help ="number of samples to create when not using dictionary:default=100000")
    args = parser.parse_args()
    main(args)
    
    