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
sys.path.append("../")

from coreLib.utils import create_dir,LOG_INFO
from coreLib.ops import images2words,cleanRecogDataset,createRecogTrainingDataset


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
    data_dim    =   int(args.data_dim)
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
    df_hand,df_dict=cleanRecogDataset(df=dataset)

    LOG_INFO("Creating Recognizer Training Dataset")
    createRecogTrainingDataset(df_hand=df_hand,
                               df_dict=df_dict,
                               img_height=img_height,
                               data_dim=data_dim,
                               raw_path=raw_path,
                               images2words_path=images2words_path,
                               save_path=save_path,
                               num_samples_dict=int(args.num_samples),
                               total_dict=int(args.total_dict))
                               
    
               
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the data folder that contains converted and raw folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=128,help ="fixed height for each grapheme: default=128")
    parser.add_argument("--data_dim",required=False,default=256,help ="dimension of word images: default=256")
    parser.add_argument("--num_samples_dict",required=False,default=1,help ="number of samples to create per dictionary word: default=5")
    parser.add_argument("--total_dict",required=False,default=100000,help ="the total number of words to take from dict: default=20000")
    args = parser.parse_args()
    main(args)
    
    