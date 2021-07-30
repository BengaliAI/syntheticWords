# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
# @TODO: 
# add mask generation while storing
# handle data.csv

from __future__ import print_function
from coreLib.utils import LOG_INFO
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os
import tensorflow as tf
from tqdm.auto import tqdm
import numpy as np 
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#---------------------------------------------------------------
def toTfrecord(df,rnum,rec_path,mask_dim):
    tfrecord_name=f'{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in range(len(df)):
            img_path=df.iloc[idx,0]
            glabel  =df.iloc[idx,1]
            clabel  =df.iloc[idx,2]
            imgw    =df.iloc[idx,3]
            tgtw    =df.iloc[idx,4]    
             
            tgt_path=img_path.replace("images","targets")
            # img
            with(open(img_path,'rb')) as fid:
                image_png_bytes=fid.read()

            # tgt
            with(open(tgt_path,'rb')) as fid:
                target_png_bytes=fid.read()


            # mask
            # img
            imask=np.zeros(mask_dim)
            imask[:,:imgw]=1
            imask=imask.flatten().tolist()
            imask=[int(i) for i in imask]
            # tgt
            tmask=np.zeros(mask_dim)
            tmask[:,:tgtw]=1
            tmask=tmask.flatten().tolist()
            tmask=[int(i) for i in tmask]
            # feature desc
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'target':_bytes_feature(target_png_bytes),
                    'clabel':_int64_list_feature(clabel),
                    'glabel':_int64_list_feature(glabel),
                    'img_mask':_int64_list_feature(imask),
                    'tgt_mask':_int64_list_feature(tmask)
            }
            
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)  
            
def df2rec(df,save_path,data_size,mask_dim):
    '''
        tf record wrapper
        args:
            df          :   a dataframe with the following columns
                                "img_path","glabel","clabel","image_mask","target_mask"
            save_path   :   path to save the tfrecords
            data_size   :   how many data to store in a single record
            mask_dim    :   create image and target masks
    '''
    LOG_INFO(f"Creating TFRECORDS:{save_path}")
    for idx in tqdm(range(0,len(df),data_size)):
        _df        =   df.iloc[idx:idx+data_size]  
        rnum       =   idx//data_size
        toTfrecord(_df,rnum,save_path,mask_dim)