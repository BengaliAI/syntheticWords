#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import os 
import json
import pandas as pd
from tqdm import tqdm
import sys 

from coreLib.utils import create_dir,LOG_INFO
from coreLib.ops import images2words,cleanRecogDataset,createRecogDataset

#--------------------
# main
#--------------------
def main(args):
    '''
        * Creates a images2words data
        * cleans up the data
        * creates a dataset for Recognizer Training
    '''  
    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    # ops
    converted_path = os.path.join(data_path,'converted')
    raw_path       = os.path.join(data_path,'RAW')
    # error check
    LOG_INFO("checking args error")
    if not os.path.exists(raw_path):
        raise ValueError("Wrong Data directory given. No RAW png folder in data path")
    if not os.path.exists(converted_path):
        raise ValueError("Wrong Data directory given. No converted folder in data path")
    
    LOG_INFO("Creating images 2 words")
    dataset,images2words_path=images2words(converted_path=converted_path,
                                           save_path=save_path)
    LOG_INFO("cleaning lexicon-based dataset")
    dataset=cleanRecogDataset(dataset=dataset)
    
    LOG_INFO("Creating Recognizer Training Dataset")
    createRecogDataset( dataset=dataset,
                        img_height=img_height,
                        img_width=img_width,
                        raw_path=raw_path,
                        images2words_path=images2words_path,
                        save_path=save_path,
                        num_samples=int(args.num_samples))
                        
    
               
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the data folder that contains converted and raw folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=32,help ="height for each grapheme: default=32")
    parser.add_argument("--img_width",required=False,default=128,help ="width dimension of word images: default=128")
    parser.add_argument("--num_samples",required=False,default=10,help ="number of samples to create per word: default=10")
    args = parser.parse_args()
    main(args)
    
    