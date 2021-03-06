#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored
from tqdm import tqdm
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
import argparse
import os
import sys
import cv2 
sys.path.append("../")
import pandas as pd
import h5py
import numpy as np 
from coreLib.utils import LOG_INFO,create_dir
from glob import glob
from tqdm import tqdm


sys.path.append('..')
# ---------------------------------------------------------
# globals
# ---------------------------------------------------------
# number of images to store in a h5file
DATA_NUM  = 1024
DS        = pd.read_csv(os.path.join(os.getcwd(),"resources","dataset.csv"))
DS        = DS.rename(columns={"Unnamed: 0":'iden'})
DS.to_csv(os.path.join(os.getcwd(),"resources","dataset.csv"),index=False)
#---------------------------------------------------------------
def get_images_and_labels(image_paths):
    '''
        returns stacked images and labels
        args:
            image_paths  : paths of the images to store
    '''
    # get images
    images=[np.expand_dims(cv2.imread(image_path),axis=0) for image_path in image_paths]
    images=np.vstack(images)
    # get labels
    idens =[os.path.basename(image_path) for image_path in image_paths]
    labels=[DS.iden.loc[DS.image==iden].tolist()[0] for iden in idens]
    labels=np.vstack(labels)
    return images,labels
    
def to_h5(image_paths,
          save_dir,
          rec_num):
    '''
        creates a h5 store 
        args:
            image_paths  : paths of the images to store
            save_dir     : directory to store the data
            rec_num      : the number of h5 file

    '''
    images,labels=get_images_and_labels(image_paths)
    # Create a new HDF5 file
    file = h5py.File(os.path.join(save_dir,f"{rec_num}.h5"), "w")

    # Create a dataset in the file
    dataset = file.create_dataset("images", np.shape(images), h5py.h5t.STD_U8BE, data=images)
    meta_set = file.create_dataset("meta", np.shape(labels), data=labels)
    file.close()

def genH5s(_paths,save_dir):
    '''	        
        tf record wrapper
        args:	        
            _paths    :   all image paths for a mode	        
            mode_dir  :   location to save the tfrecords	    
    '''
    for i in tqdm(range(0,len(_paths),DATA_NUM)):
        # paths
        image_paths= _paths[i:i+DATA_NUM]
        # record num
        r_num=i // DATA_NUM
        # create tfrecord
        to_h5(image_paths,save_dir,r_num)    

# ---------------------------------------------------------
def main(args):
    '''
        this creates the h5 store with test and train data
    '''
    data_dir = args.data_dir
    # path of images dir
    train_img_dir  = os.path.join(data_dir,"dict")
    test_img_dir   = os.path.join(data_dir,"hand")
    
    # paths    
    train_img_paths=[img_path for img_path in tqdm(glob(os.path.join(train_img_dir,"*.*")))]
    test_img_paths =[img_path for img_path in tqdm(glob(os.path.join(test_img_dir,"*.*")))]
    
    # h5 saving directories 
    save_dir    = args.save_dir
    rec_dir     = create_dir(save_dir,'h5store')
    train_save  = create_dir(rec_dir,'train')
    test_save   = create_dir(rec_dir,'test')
      
    
    # h5 store
    LOG_INFO("Creating test h5 stores")
    genH5s(test_img_paths,test_save)
    LOG_INFO("Creating train h5 stores")
    genH5s(train_img_paths,train_save)
    
# ---------------------------------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='script to create h5store for reconizer training')
    parser.add_argument('data_dir',help="The path to the folder that contains dict and hand images") 
    parser.add_argument('save_dir',help="The path to the folder where the h5 stores will be saved") 
    args = parser.parse_args()
    main(args)