#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd
import random
import cv2
import numpy as np
import argparse
import sys
from tqdm.auto import tqdm
from ast import literal_eval
from glob import glob
from coreLib.utils import * 
from coreLib.words import getRandomSyntheticData

tqdm.pandas()
# globas to store problematic data
NOT_FOUND   =  []

#--------------------
# helper functions
#--------------------

def drop_empty_list(x):
    '''
        returns nan if x is empty
    '''
    if len(x)==0:
        return np.nan
    elif x=='[]':
        return np.nan
    else:
        return x
#--------------------
# main
#--------------------


def main(args):
    '''
        creates the images based on labels
    '''
    data_path  =  args.data_path                        
    label_csv  =  os.path.join(data_path,'label.csv')
    png_dir    =  os.path.join(data_path,'RAW')
    lexicon_csv=  os.path.join(data_path,'indicword_lexicon_grapheme.csv')
    # error check
    LOG_INFO("checking args error")
    if not os.path.exists(png_dir):
        raise ValueError("Wrong Data directory given. No RAW png folder in data path")
    
    try:
        df=pd.read_csv(label_csv)
    except Exception as e:
        LOG_INFO(f"Error While reading label.csv:{e}",mcolor='red')    
        sys.exit(1)


    try:
        df=pd.read_csv(lexicon_csv)
    except Exception as e:
        LOG_INFO(f"Error While reading indicword_lexicon_grapheme.csv:{e}",mcolor='red')    
        sys.exit(1)

    LOG_INFO("loading args")
    base_dir   =  create_dir(data_path,args.save_iden)  
    # images per lexicon
    num_total  =  int(args.sample_num)
    # Fixed height (without any synthetic correction)
    img_height =  int(args.img_height)
    data_dim   =  int(args.data_dim)  
    
    # read data
    label_df     =  pd.read_csv(label_csv)
    lexicon_df   =  pd.read_csv(lexicon_csv)
    # idx clean-up
    lexicon_df=lexicon_df[['lexicon', 'graphemes']]
    # drop duplicates
    lexicon_df=lexicon_df.drop_duplicates()
    # convert string list to list
    lexicon_df.graphemes=lexicon_df.graphemes.progress_apply(lambda x: literal_eval(x))
    #drop empty
    lexicon_df.graphemes=lexicon_df.graphemes.progress_apply(lambda x: drop_empty_list(x))
    lexicon_df.dropna(inplace=True)
            
    for lex,data in tqdm(zip(lexicon_df.lexicon,lexicon_df.graphemes),total=len(lexicon_df)):
        # create a label dir
        base_dir=create_dir(base_dir,lex)
        LOG_INFO(lex)                
        # save images from graphemes
        for i in tqdm(range(num_total)):
            _path=os.path.join(base_dir,f'{i}.png')
            # only create a new image if the image is non-existant
            if not os.path.exists(_path):
                img=getRandomSyntheticData(grapheme_list=data,
                                           label_df=label_df,
                                           png_dir=png_dir,
                                           img_height=img_height,
                                           data_dim=data_dim )

                # if a not found grapheme is returned append to list
                if len(img)==2 and img[0]==None:
                    LOG_INFO(f"Grapheme Does Not Exist:{img[1]} for: {lex},{data}")
                    NOT_FOUND.append(img[1])
                    break
                else:

                    if img is None:
                        break
                    else:
                            cv2.imwrite(_path,img)
            else:
                break
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("synthetic word generation script")
    parser.add_argument("data_path", help="Path of the data folder that contains label.csv,RAW folder and indicword_lexicon_grapheme.csv")
    parser.add_argument("save_iden", help="identifier of the folfer to save data")
    parser.add_argument("--sample_num",required=False,default=1000,help = "number of samples to create : default=1000")
    parser.add_argument("--img_height",required=False,default=128,help ="fixed height for each grapheme: default=128")
    parser.add_argument("--data_dim",required=False,default=256,help ="dimension of word images: default=256")
    args = parser.parse_args()
    main(args)
    
    