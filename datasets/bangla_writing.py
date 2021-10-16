# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
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

def main(args):
    data_path=args.data_path
    save_path=args.save_path
    main_path=create_dir(save_path,"bw")
    save_path=create_dir(main_path,"images")
    
    filepath=[]
    word=[]
    source=[]
    i=0
    LOG_INFO(save_path)
    # get image paths
    img_paths=[img_path for img_path in glob(os.path.join(data_path,"*.jpg"))]
    # iterate
    for img_path in tqdm(img_paths):
        # extract images and labels
        imgs,labels=extract_word_images_and_labels(img_path)
        if len(imgs)>0:
            for img,label in zip(imgs,labels):
                try:
                    # thresh
                    _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    img=stripPads(img,255)
                    img_save_path=os.path.join(save_path,f"{i}.png")
                    # save
                    cv2.imwrite(img_save_path,img)
                    # append
                    filepath.append(img_save_path)
                    word.append(label)
                    source.append(os.path.basename(img_path))
                    i=i+1
                except Exception as e: 
                    LOG_INFO(f"error in creating image:{img_path} label:{label},error:{e}",mcolor='red')

    df=pd.DataFrame({"filepath":filepath,"word":word,"source":source})
    df.to_csv(os.path.join(main_path,"data.csv"),index=False)

#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Bangla Writting Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the converted folder that contains .jsons and .jpg files")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    args = parser.parse_args()
    main(args)

