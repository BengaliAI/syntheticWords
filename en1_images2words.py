#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author: Tahsin Reasat, MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import json
import cv2
import numpy as np
import sys 
import argparse
import string
from glob import glob
from tqdm import tqdm
from coreLib.utils import *

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
print(SYMBOLS)
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
# main
#--------------------


def main(args):
    '''
        creates the images based on labels
    '''
    # data path
    data_path="/media/ansary/DriveData/Work/bengalAI/datasets/BanglaWords/"
    # save path
    save_path="/media/ansary/DriveData/Work/bengalAI/datasets/"
    save_path=create_dir(save_path,'words')
    try:
        # get image paths
        img_paths=[img_path for img_path in glob(os.path.join(data_path,"*.jpg"))]
        # iterate
        for img_path in tqdm(img_paths):
            # extract images and labels
            imgs,labels=extract_word_images_and_labels(img_path)
            if len(imgs)>0:
                for img,label in zip(imgs,labels):
                    # label path
                    label_path=create_dir(save_path,label)
                    # count of existing images
                    label_iden=len([_path for _path in glob(os.path.join(label_path,"*.*"))]) 
                    # save path for the word
                    img_save_path=os.path.join(label_path,f"{label_iden}.png")
                    try:
                        # save
                        cv2.imwrite(img_save_path,img)
                    except Exception as e: 
                        LOG_INFO(f"error in creating image:{img_path} label:{label},error:{e}",mcolor='red')
            
    except Exception as e:
        LOG_INFO(f"Error While processing:{e}",mcolor='red')    
        sys.exit(1)

        
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("image to word generation script for banglawritting dataset")
    parser.add_argument("data_path", help="Path of the data folder that contains .jpg s and .json s")
    parser.add_argument("save_path", help="Path of the directory to save the images per their labels")
    args = parser.parse_args()
    main(args)
    
    