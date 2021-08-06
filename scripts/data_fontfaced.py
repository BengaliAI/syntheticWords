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
from coreLib.synthetic import createFontFacedWords
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    num_samples =   int(args.num_samples)
    dict_max_len=   int(args.dict_max_len)
    # dataset object
    ds=DataSet(data_dir=data_path,check_english=True)
    LOG_INFO("Creating english fontfaced data")
    createFontFacedWords(iden="english.fontfaced",
                        save_dir=save_path,
                        all_fonts=ds.english.all_fonts,
                        font_path=ds.english.font,
                        img_dim=(img_height,img_width),
                        comp_dim=img_height,
                        valid_graphemes=ds.english.ffvocab,
                        num_samples=num_samples,
                        dict_max_len=dict_max_len)
    
    ds=DataSet(data_dir=data_path,check_english=False)
    LOG_INFO("Creating bangla fontfaced data")
    createFontFacedWords(iden="bangla.fontfaced",
                        save_dir=save_path,
                        all_fonts=ds.bangla.all_fonts,
                        font_path=ds.bangla.font,
                        img_dim=(img_height,img_width),
                        comp_dim=img_height,
                        valid_graphemes=ds.bangla.ffvocab,
                        num_samples=num_samples,
                        dict_max_len=dict_max_len)
    

    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training: english Synthetic (numbers and graphemes) Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the source data folder ")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension of word images: default=512")
    parser.add_argument("--num_samples",required=False,default=100000,help ="number of samples to create when not using dictionary:default=100000")
    parser.add_argument("--dict_max_len",required=False,default=20,help ="max number of graphemes in a word:default=20")
    
    args = parser.parse_args()
    main(args)
    
    