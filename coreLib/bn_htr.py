#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os 
import random
import cv2 
import numpy as np 
import pandas as pd 
from .utils import LOG_INFO, correctPadding,GraphemeParser
from tqdm import tqdm
tqdm.pandas()
GP=GraphemeParser()
#---------------------------------------------------------------
# ops
#---------------------------------------------------------------
def processData_BN_HTR(ds,dim):
    '''
        creates the training data from **BN HTR Dataset**
        args:
            ds            :  dataset object
            dim           :  (img_height,img_width) tuple to resize to 
    '''
    # properly set height and width
    img_height,img_width=dim

    img_idens=[]
    img_labels=[]
    
    # data
    df=ds.bn_htr.df
    df["img_path"]=df.filename.progress_apply(lambda x: os.path.join(ds.bn_htr.dir,x))

    save_path=ds.bn_htr_path 
    

    for idx in tqdm(range(len(df))):
        img_path=df.iloc[idx,2]
        labels=df.iloc[idx,1]
        try:
            # image and label
            img=cv2.imread(img_path,0)
            # resize (heigh based)
            h,w=img.shape 
            width= int(img_height* w/h) 
            img=cv2.resize(img,(width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            # save
            img=correctPadding(img,dim)     
            img_save_path=os.path.join(save_path,f"{idx}.png")
            
            
            # save
            cv2.imwrite(img_save_path,img)
            # append
            img_idens.append(f"{idx}.png")
            img_labels.append("".join(labels))

        except Exception as e:
            LOG_INFO(e)
            LOG_INFO(img_path)    

    # test dataframe
    df              =   pd.DataFrame({"filename":img_idens,"word":img_labels})
    # graphemes
    df["graphemes"] =   df.word.progress_apply(lambda x:GP.word2grapheme(x))
    # unicodes
    df["unicodes"]  =   df.word.progress_apply(lambda x:[i for i in x])
    
    df.dropna(inplace=True)
    
    

    df=df[["filename","word","graphemes","unicodes"]]
    df.to_csv(ds.bn_htr_csv,index=False)
