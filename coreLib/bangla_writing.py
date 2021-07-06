# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import json
import cv2
import numpy as np
import pandas as pd 
import string
import random
from glob import glob
from tqdm import tqdm
from .utils import stripPads,correctPadding,LOG_INFO,GraphemeParser
tqdm.pandas()
#--------------------
# GLOBALS
#--------------------
# symbols to avoid 
SYMBOLS = ['`','~','!','@','#','$','%',
           '^','&','*','(',')','_','-',
           '+','=','{','[','}','}','|',
           '\\',':',';','"',"'",'<',
           ',','>','.','?','/',
           '১','২','৩','৪','৫','৬','৭','৮','৯','০',
           '।']
SYMBOLS+=list(string.ascii_letters)
SYMBOLS+=[str(i) for i in range(10)]
GP=GraphemeParser()
#--------------------------------images2words------------------------------------------------------------
#--------------------
# helper functions
#--------------------

def extract_word_images_and_labels(img_path):
    '''
        extracts word images and labels from a given image
        args:
            img_path : path of the image
        returns:
            (images,labels)
            list of images and labels
    '''
    imgs=[]
    labels=[]
    # json_path
    json_path=img_path.replace("jpg","json")
    # read image
    data=cv2.imread(img_path,0)
    # label
    label_json = json.load(open(json_path,'r'))
    # get word idx
    for idx in range(len(label_json['shapes'])):
        # label
        label=str(label_json['shapes'][idx]['label'])
        # special charecter negation
        if not any(substring in label for substring in SYMBOLS):
            labels.append(label)
            # crop bbox
            xy=label_json['shapes'][idx]['points']
            # crop points
            x1 = int(np.round(xy[0][0]))
            y1 = int(np.round(xy[0][1]))
            x2 = int(np.round(xy[1][0]))
            y2 = int(np.round(xy[1][1]))
            # image
            img=data[y1:y2,x1:x2]
            imgs.append(img)
    return imgs,labels

    
#--------------------
# ops
#--------------------
def processData_BANGLA_WRITING(ds,dim):
    '''
        creates the testing data from **Bangla Writing** dataset
        args:
            ds            :  dataset object
            dim           :  (img_height,img_width) tuple to resize to 
    '''
    # properly set height and width
    img_height,img_width=dim

    img_idens=[]
    img_labels=[]
    
    
    i=0
    
    
    save_path=ds.bangla_writing_path
    LOG_INFO(save_path)
    
    
    # get image paths
    img_paths=[img_path for img_path in glob(os.path.join(ds.pages,"*.jpg"))]
    # iterate
    for img_path in tqdm(img_paths):
        # extract images and labels
        imgs,labels=extract_word_images_and_labels(img_path)
        if len(imgs)>0:
            for img,label in zip(imgs,labels):
                try:
                    
                    # thresh
                    blur = cv2.GaussianBlur(img,(5,5),0)
                    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    img=stripPads(img,255)
                    # resize (heigh based)
                    h,w=img.shape 
                    width= int(img_height* w/h) 
                    img=cv2.resize(img,(width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
                    # save
                    img=correctPadding(img,dim)     
                    img_save_path=os.path.join(save_path,f"{i}.png")
                    # save
                    cv2.imwrite(img_save_path,img)
                    # append
                    img_idens.append(f"{i}.png")
                    img_labels.append(label)
                    i=i+1
                    
                except Exception as e: 
                    LOG_INFO(f"error in creating image:{img_path} label:{label},error:{e}",mcolor='red')
    
    
    
    # test dataframe
    df              =   pd.DataFrame({"filename":img_idens,"word":img_labels})
    # graphemes
    df["graphemes"] =   df.word.progress_apply(lambda x:GP.word2grapheme(x))
    # unicodes
    df["unicodes"]  =   df.word.progress_apply(lambda x:[i for i in x])
    
    df.dropna(inplace=True)
    
    

    df=df[["filename","word","graphemes","unicodes"]]
    df.to_csv(ds.bangla_writing_csv,index=False)    