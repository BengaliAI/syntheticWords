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
import cv2
from tqdm import tqdm
from ast import literal_eval
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 
from coreLib.utils import LOG_INFO, create_dir,correctPadding,extend_vocab,WordCleaner
from coreLib.dataset import bangla
tqdm.pandas()
#--------------------
# globals
#--------------------
vocab_json  ="../vocab.json"
WC=WordCleaner()
garbage=["া","্বা","্ল","ভঁে"]
#--------------------
# helpers
#--------------------
def process(df,img_dir,img_dim,save,ptype):
    '''
        process a specific dataframe with filename,word,graphemes and mode
        args:
            df      :   the dataframe to process
            img_dir :   image directory that holds all the images
            img_dim :   tuple of (img_height,img_width)
            save    :   saving resource paths    
    '''
    filename=[]
    labels=[]
    imasks=[]
    comp_dim,_=img_dim
    for idx in tqdm(range(len(df))):
        try:
            fname       =   df.iloc[idx,0]
            comps       =   df.iloc[idx,2]
            # create img
            img_path=os.path.join(img_dir,fname)
            img=cv2.imread(img_path,0)
            # resize (heigh based)
            h,w=img.shape 
            width= int(comp_dim* w/h) 
            img=cv2.resize(img,(width,comp_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            # correct padding
            img,imask=correctPadding(img,img_dim,ptype=ptype)
            # save
            fname=f"{idx}.png"
            cv2.imwrite(os.path.join(save.img,fname),img)
            filename.append(fname)
            labels.append(comps)
            imasks.append(imask)
            
        except Exception as e:
            LOG_INFO(e)
    df=pd.DataFrame({"filename":filename,"labels":labels,"image_mask":imasks})
    df.to_csv(os.path.join(save.csv),index=False)
        
#--------------------
# main
#--------------------
def main(args):
    '''
        src
        ** process data.csv
        ** train test 
        save
    '''

    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    iden        =   args.iden
    save        =   create_dir(save_path,iden)
    img_dim     =   (img_height,img_width)
    ptype       =   args.ptype
    #--------------------
    # src
    #--------------------
    def drop_garbage(x):
        if len(list(set(x) & set(garbage)))>0:
            return None
        else:
            return x

    class src:
        data_csv    =   os.path.join(data_path,"data.csv")
        img_dir     =   os.path.join(data_path,"images")
        # process data_csv
        df=pd.read_csv(data_csv)
        df.word=df.word.progress_apply(lambda x: WC.clean(x))
        df.dropna(inplace=True)
        df.word=df.word.progress_apply(lambda x: x if  set([i for i in x]).issubset(bangla.valid) else None)
        df.dropna(inplace=True)
        
        df.graphemes=df.graphemes.progress_apply(lambda x: literal_eval(x))
        df.graphemes=df.graphemes.progress_apply(lambda x: drop_garbage(x))
        df.dropna(inplace=True)
        # train test
        train=df.loc[df["mode"]=="train"]
        test =df.loc[df["mode"]=="test"]
        
    #--------------------
    # save
    #--------------------

    class train:
        dir=create_dir(save,"train")
        img=create_dir(dir,"images")
        csv=os.path.join(dir,"data.csv")
        #filename,labels,image_mask,target_mask
    class test:
        dir=create_dir(save,"test")
        img=create_dir(dir,"images")
        csv=os.path.join(dir,"data.csv")
    
    with open(vocab_json) as f:
        curr_vocab = json.load(f)["gvocab"]
    
    new_vocab=extend_vocab(src.df.graphemes.tolist(),curr_vocab)
    # config 
    config={'gvocab':curr_vocab+new_vocab}
    with open(vocab_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)

    LOG_INFO("Test")
    process(src.test,src.img_dir,img_dim,test,ptype)
    LOG_INFO("Train")
    process(src.train,src.img_dir,img_dim,train,ptype)

#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Processing Dataset Script")
    parser.add_argument("data_path", help="Path of the source data folder for any naturally writen images/data.csv pair dataset ")
    parser.add_argument("save_path", help="Path of the directory to save the  processed dataset")
    parser.add_argument("iden", help="identifier of the dataset")
    parser.add_argument("--img_height",required=False,default=64,help ="height for each grapheme: default=64")
    parser.add_argument("--img_width",required=False,default=512,help ="width dimension of word images: default=512")
    parser.add_argument("--ptype",required=False,default="left",help ="type of padding to use(for CRNN use central , for ROBUSTSCANNER use left): default=left")
    args = parser.parse_args()
    main(args)