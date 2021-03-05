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
import random
import tensorflow as tf 
import json
import sys 
sys.path.append("../")
from coreLib.utils import LOG_INFO,create_dir
from glob import glob
from tqdm import tqdm


sys.path.append('..')
# ---------------------------------------------------------
# globals
# ---------------------------------------------------------

# number of images to store in a tfrecord
DATA_NUM  = 1024

with open(os.path.join(os.getcwd(),'resources','dataset.json'), 'r') as fp:
    MAP_DICT = json.load(fp)
    assert type(MAP_DICT)==dict,"Houston we've got a problem" 
#---------------------------------------------------------------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def to_tfrecord(image_paths,save_dir,r_num):
    '''	            
      Creates tfrecords from Provided Image Paths	        
      args:	        
        image_paths     :   specific number of image paths	       
        save_dir        :   location to save the tfrecords	           
        r_num           :   record number	
    '''
    # record name
    tfrecord_name='{}.tfrecord'.format(r_num)
    # path
    tfrecord_path=os.path.join(save_dir,tfrecord_name)
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        for image_path in image_paths:
            
            target_path=str(image_path).replace('images','targets')
            #image
            with(open(image_path,'rb')) as fid:
                image_bytes=fid.read()
            # target
            with(open(target_path,'rb')) as fid:
                target_bytes=fid.read()
            #label
            label=MAP_DICT[os.path.basename(image_path)]
            
            data ={ 'image':_bytes_feature(image_bytes),
                    'target':_bytes_feature(target_bytes),
                    'label':_bytes_feature(bytes(label,encoding= 'utf-8'))
            }
            # write
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)


def genTFRecords(_paths,mode_dir):
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
        to_tfrecord(image_paths,mode_dir,r_num)    

# ---------------------------------------------------------
def main(args):
    '''
        this creates the tfrecords for the images and targets
    '''
    data_dir = args.data_dir
    # path of images dir
    img_dir  = os.path.join(data_dir,"images")
    
    # paths    
    img_paths=[img_path for img_path in glob(os.path.join(img_dir,"*.*"))]
    
    # tfrecord saving directories 
    save_dir = args.save_dir
    rec_dir  = create_dir(save_dir,'tfrecords')
    
    # tfrecords
    genTFRecords(img_paths,rec_dir)
    
# ---------------------------------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='script to create synthetic tfrecords data for style transfer')
    parser.add_argument('data_dir',help="The path to the folder that contains images and targets") 
    parser.add_argument('save_dir',help="The path to the folder where the tfrecords will be saved") 
    args = parser.parse_args()
    main(args)