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
import math
import pandas as pd 
import cv2
from tqdm import tqdm
from ast import literal_eval
from coreLib.utils import LOG_INFO, create_dir,get_encoded_label,pad_encoded_label
from coreLib.store import df2rec
tqdm.pandas()
#--------------------
# globals
#--------------------
vocab_json  ="../vocab.json"
with open(vocab_json) as f:
    gvocab = json.load(f)["gvocab"]
    
cvocab=[]
for g in gvocab:
    for i in g:
        if i not in cvocab:
            cvocab.append(i)

cvocab=sorted(cvocab)

max_glen=10
max_clen=20
factor  =32
tf_size =1024        
#--------------------
# main
#--------------------
def main(args):
  
    data_path   =   args.data_path
    save_path   =   args.save_path
    iden        =   args.iden
    img_width   =   int(args.img_width)
    img_height  =   int(args.img_height)
    
    
    save_path        =   create_dir(save_path,"tfrecords")
    save_path        =   create_dir(save_path,iden)
    #--------------------
    # src
    #--------------------    
    csv    =   os.path.join(data_path,"data.csv")
    img    =   os.path.join(data_path,"images")
    tgt    =   os.path.join(data_path,"targets")
    # process data_csv
    df=pd.read_csv(csv)
    # literal eval
    df.labels=df.labels.progress_apply(lambda x: literal_eval(x))
    # chars
    df["chars"]=df.labels.progress_apply(lambda x: [i for i in "".join(x)])
    # img paths
    df["img_path"]=df.filename.progress_apply(lambda x:os.path.join(img,x))
    # lengths
    df["lens"]=df.labels.progress_apply(lambda x:[len(x),len([i for i in "".join(x)])])
    df["lens"]=df.lens.progress_apply(lambda x:x if x[0]<=max_glen and x[1]<=max_clen else None)
    df.dropna(inplace=True)
    # glabel clabel
    df["glabel"]=df.labels.progress_apply(lambda x: get_encoded_label(x,gvocab))
    df["clabel"]=df.chars.progress_apply(lambda x: get_encoded_label(x,cvocab))
    df["glabel"]=df.glabel.progress_apply(lambda x: pad_encoded_label(x,max_glen,0))
    df["clabel"]=df.clabel.progress_apply(lambda x: pad_encoded_label(x,max_clen,0))
    df=df[["img_path","glabel","clabel","image_mask","target_mask"]]
    # mask
    df["image_mask"]=df["image_mask"].progress_apply(lambda x:x if x > 0 else img_width)
    df["target_mask"]=df["target_mask"].progress_apply(lambda x:x if x > 0 else img_width)
      
    df["image_mask"]=df["image_mask"].progress_apply(lambda x: math.ceil((x/img_width)*(img_width//factor)))
    df["target_mask"]=df["target_mask"].progress_apply(lambda x: math.ceil((x/img_width)*(img_width//factor)))

    mask_dim=(img_height//factor,img_width//factor)
    #--------------------
    # save
    #--------------------
    df2rec(df,save_path,tf_size,mask_dim)
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Script for Creating tfrecords")
    parser.add_argument("data_path", help="Path of the processed data folder . Should hold images,targets and data.csv ")
    parser.add_argument("save_path", help="Path of the directory to save tfrecords")
    parser.add_argument("iden", help="identifier of the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension of word images: default=512")
    
    args = parser.parse_args()
    main(args)
    
    