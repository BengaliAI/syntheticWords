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
    ds=DataSet(data_dir=data_path,check_english=True)
    LOG_INFO("Creating english graphemes data")
    createWords(iden="english.graphemes",
                df=ds.english.graphemes.df,
                img_dir=ds.english.graphemes.dir,
                save_dir=save_path,
                font_path=ds.english.font,
                img_dim=(img_height,img_width),
                comp_dim=img_height,
                pad_height=pad_height,
                top_exts=[],
                bot_exts=[],
                dictionary=ds.english.dictionary,
                num_samples=num_samples)
    LOG_INFO("Creating english numbers data")
    createWords(iden="english.numbers",
                df=ds.english.numbers.df,
                img_dir=ds.english.numbers.dir,
                save_dir=save_path,
                font_path=ds.english.font,
                img_dim=(img_height,img_width),
                comp_dim=img_height,
                pad_height=pad_height,
                top_exts=[],
                bot_exts=[],
                dictionary=None,
                valid_graphemes=ds.english.number_values,
                num_samples=num_samples)            


    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training: english Synthetic (numbers and graphemes) Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the source data folder ")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--pad_height",required=False,default=20,help ="pad height for each grapheme for alignment correction: default=20")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension of word images: default=512")
    parser.add_argument("--num_samples",required=False,default=100000,help ="number of samples to create when not using dictionary:default=100000")
    args = parser.parse_args()
    main(args)
    
    