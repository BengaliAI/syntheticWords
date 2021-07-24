# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
from __future__ import print_function
from coreLib.utils import LOG_INFO
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os
import tensorflow as tf
from tqdm.auto import tqdm

#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#---------------------------------------------------------------
def toTfrecord(df,rnum,rec_path):
    '''
    Creates tfrecords from dataframe:
    * contains img_path,clabel,glabel
    '''
    tfrecord_name=f'{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in range(len(df)):
            img_path=df.iloc[idx,0]
            clabel  =df.iloc[idx,1]
            glabel  =df.iloc[idx,2]
             
            tgt_path=img_path.replace("images","targets")
            # img
            with(open(img_path,'rb')) as fid:
                image_png_bytes=fid.read()

            # tgt
            with(open(tgt_path,'rb')) as fid:
                target_png_bytes=fid.read()

            # feature desc
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'target':_bytes_feature(target_png_bytes),
                    'clabel':_int64_list_feature(clabel),
                    'glabel':_int64_list_feature(glabel),
            }
            
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)  
            
def df2rec(df,save_path,data_size):
    '''
        tf record wrapper
    '''
    LOG_INFO(f"Creating TFRECORDS:{save_path}")
    for idx in tqdm(range(0,len(df),data_size)):
        _df        =   df.iloc[idx:idx+data_size]  
        rnum       =   idx//data_size
        toTfrecord(_df,rnum,save_path)