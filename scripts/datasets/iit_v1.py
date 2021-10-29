# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
from re import S
import sys
sys.path.append('../')
import os 
import json
import cv2
import numpy as np
import pandas as pd 
import argparse
import random
from glob import glob
from tqdm.auto import tqdm
from coreLib.utils import *
random.seed(42)


def process(txt,dir,save_path,idx):
    filepath=[]
    word=[]
    source=[]
    

    with open(txt,"r") as f:
        data=f.readlines()
    
    for line in tqdm(data):
        try:
            img_path,label=line.split(" ")
            img_path=os.path.join(dir,img_path)
            img=cv2.imread(img_path,0)
            # thresh
            _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            img=stripPads(img,255)
            fname=f"{idx}.png"
            img_save_path=os.path.join(save_path,fname)
            # save
            cv2.imwrite(img_save_path,img)
            # append
            filepath.append(img_save_path)
            word.append(label)
            source.append(os.path.basename(img_path))
            idx=idx+1
        except Exception as e: 
            LOG_INFO(f"error in creating image:{img_path} label:{label},error:{e}",mcolor='red')
    df=pd.DataFrame({"filepath":filepath,"word":word,"source":source})
    return df,idx     
    


def main(args):
    vocab_path=args.vocab_path
    save_path =args.save_path
    iden      =args.iden
    
    # paths
    data_path=os.path.dirname(vocab_path)
    train_txt =os.path.join(data_path,"train.txt")
    val_txt   =os.path.join(data_path,"val.txt")
    test_txt  =os.path.join(data_path,"test.txt")
    
    assert os.path.exists(train_txt),"Train text missing"
    assert os.path.exists(val_txt),"Val text missing"
    assert os.path.exists(test_txt),"Test text missing"

    if iden is None:
        iden=os.path.basename(data_path)
        iden=f"iit.{iden}"


    main_path=create_dir(save_path,f"{iden}")
    save_path=create_dir(main_path,"images")
    
    LOG_INFO(save_path)
    
    # process
    dfs=[]
    df,idx=process(train_txt,data_path,save_path,idx=0)
    dfs.append(df)
    df,idx=process(val_txt,data_path,save_path,idx)
    dfs.append(df)    
    df,idx=process(test_txt,data_path,save_path,idx)
    dfs.append(df)
    df=pd.concat(dfs,ignore_index=True)
    df.to_csv(os.path.join(main_path,"data.csv"),index=False)

#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("iit indic v1 Dataset Creation Script")
    parser.add_argument("vocab_path", help="Path of the lexicon.txt file")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--iden",required=False,default=None,help="identifier to identify the dataset")
    args = parser.parse_args()
    main(args)

