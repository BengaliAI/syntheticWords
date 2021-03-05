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
from coreLib.ops import images2words,cleanDataset,createDataset


#--------------------
# main
#--------------------
def main(args):
    '''
        * Creates a images2words data
        * cleans up the data
        * creates a dataset for synthetic style transformation
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
    LOG_INFO("cleaning dataset")
    dataset=cleanDataset(df=dataset)
    # save
    dataset.to_csv(os.path.join(os.getcwd(),'resources',"dataset.csv"),index=False)
    
    LOG_INFO("Dataset creation")
    createDataset(dataset=dataset,
                  img_height=img_height,
                  data_dim=data_dim,
                  raw_path=raw_path,
                  images2words_path=images2words_path,
                  save_path=save_path)
    
    
    # map image_id to labels
    map_dict={}
    map_json=os.path.join(os.getcwd(),'resources','dataset.json')
    for iid,label in tqdm(zip(dataset.image_id.tolist(),dataset.label.tolist()),total=len(dataset)):
        map_dict[iid]= label
    # save map
    with open(map_json, 'w') as fp:
        json.dump(map_dict, fp, sort_keys=True, indent=4,ensure_ascii=False)

    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Style transfer model synthetic data generation script")
    parser.add_argument("data_path", help="Path of the data folder that contains converted and raw folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=128,help ="fixed height for each grapheme: default=128")
    parser.add_argument("--data_dim",required=False,default=256,help ="dimension of word images: default=256")
    args = parser.parse_args()
    main(args)
    
    