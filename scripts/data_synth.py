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
from coreLib.dataset import DataSet,bangla
from coreLib.synthetic import createSyntheticData
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
    pad_type    =   args.pad_type
    # dataset object
    ds=DataSet(data_dir=data_path)
    assert num_samples> len(ds.all_fonts),f"Number of samples should be >{len(ds.all_fonts)}"
    # process dictionary
    LOG_INFO("Creating bangla synthetic data")
    dictionary=pd.read_csv("../dictionary.csv")
    dictionary.graphemes=dictionary.graphemes.progress_apply(lambda x: literal_eval(x))
    dictionary.graphemes=dictionary.graphemes.progress_apply(lambda x: x if set([i for i in x]).issubset(ds.known_graphemes) else None)
    dictionary.dropna(inplace=True)
    
    
    createSyntheticData(iden="synth",
                        df=ds.graphemes.df,
                        img_dir=ds.graphemes.dir,
                        save_dir=save_path,
                        fonts=ds.all_fonts,
                        img_dim=(img_height,img_width),
                        comp_dim=img_height,
                        pad_height=pad_height,
                        top_exts=bangla.top_exts,
                        bot_exts=bangla.bot_exts,
                        dictionary=dictionary,
                        sample_per_word=num_samples,
                        pad_type=pad_type)
    

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
    parser.add_argument("--num_samples",required=False,default=100,help ="number of samples to create when using dictionary:default=100")
    parser.add_argument("--pad_type",required=False,default="left",help ="type of padding to use(for CRNN use central , for ROBUSTSCANNER use left): default=left")
    args = parser.parse_args()
    main(args)