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

from coreLib.utils import *
from coreLib.store import createRecords
tqdm.pandas()

def main(args):
    
    data_dir    =   args.data_dir
    iden        =   args.iden
    LOG_INFO(data_dir)
    LOG_INFO(iden)
    # storing
    csv=os.path.join(data_dir,"data.csv")
    LOG_INFO(csv)
    save_path=create_dir(data_dir,iden)
    LOG_INFO(save_path)
    createRecords(csv,save_path)

#-----------------------------------------------------------------------------------
if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("TFRecords Dataset Creating Script")
    parser.add_argument("data_dir", help="Path of the source data folder that contains langauge datasets")
    parser.add_argument("iden",help="identifier to identify the dataset")
    args = parser.parse_args()
    main(args)