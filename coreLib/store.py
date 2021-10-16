#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import json
import math
import pandas as pd 
import tensorflow as tf
import numpy as np 
from ast import literal_eval
from tqdm.auto import tqdm
from .utils import *
tqdm.pandas()
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
cols=["filepath","word","mask","pu_label","pg_label","pru_label","prg_label"]
eval_cols=cols[2:]
    
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def toTfrecord(df,rnum,rec_path):
    '''
        args:
            df      :   the dataframe that contains the information to store
            rnum    :   record number
            rec_path:   save_path
            mask_dim:   the dimension of the mask
    '''
    tfrecord_name=f'{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in range(len(df)):
            # base
            img_path=df.iloc[idx,0]
            # img
            with(open(img_path,'rb')) as fid:
                image_png_bytes=fid.read()
            # feature desc
            data ={ 'image':_bytes_feature(image_png_bytes)}

            for cidx,col in enumerate(cols):
                if col in eval_cols:
                    data[col]=_int64_list_feature(df.iloc[idx,cidx]) 

            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)  

def createRecords(data,save_path,tf_size=10240):
    '''
        creates tf records:
        args:
            data        :   either the csv path or a dataframe
            save_path   :   location to save tfrecords
    '''
    if type(data)=="str":
        data=pd.read_csv(data)
        for col in eval_cols:
            data[col]=data[col].progress_apply(lambda x: literal_eval(x))
    
    LOG_INFO(f"Creating TFRECORDS:{save_path}")
    for idx in tqdm(range(0,len(data),tf_size)):
        df        =   data.iloc[idx:idx+tf_size]  
        rnum      =   idx//tf_size
        toTfrecord(df,rnum,save_path)

    
    