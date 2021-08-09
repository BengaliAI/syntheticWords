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
import math
import pandas as pd 

import tensorflow as tf
import numpy as np 

from ast import literal_eval
from tqdm.auto import tqdm

from coreLib.utils import LOG_INFO, create_dir,get_encoded_label,pad_encoded_label


tqdm.pandas()

#--------------------
# globals
#--------------------
vocab_json  ="../vocab.json"
with open(vocab_json) as f:
    vocab = json.load(f)

cvocab=vocab["cvocab"]
gvocab=vocab["gvocab"]
       
LOG_INFO(f"Unicode class:{len(cvocab)}")
LOG_INFO(f"Grapheme class:{len(gvocab)}")


#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def toTfrecord(df,
              rnum,
              rec_path,
              mask_dim,
              use_font):
    '''
        args:
            df      :   the dataframe that contains the information to store
            rnum    :   record number
            rec_path:   save_path
            mask_dim:   the dimension of the mask
            use_font:   store fontfaced images 
    '''
    tfrecord_name=f'{rnum}.tfrecord'
    tfrecord_path=os.path.join(rec_path,tfrecord_name) 
    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for idx in range(len(df)):
            # base
            img_path=df.iloc[idx,0]
            glabel  =df.iloc[idx,1]
            clabel  =df.iloc[idx,2]

            # img
            with(open(img_path,'rb')) as fid:
                image_png_bytes=fid.read()

            # feature desc
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'clabel':_int64_list_feature(clabel),
                    'glabel':_int64_list_feature(glabel)
            }

            if use_font:
                tgt_path=img_path.replace("images","targets")
                # tgt
                with(open(tgt_path,'rb')) as fid:
                    target_png_bytes=fid.read()
                data['target']=_bytes_feature(target_png_bytes)
            

            if mask_dim is not None:
                imgw    =df.iloc[idx,3]
                # img mask
                imask=np.zeros(mask_dim)
                # valid true format: processing in model
                imask[:,:imgw]=1
                imask=imask.flatten().tolist()
                imask=[int(i) for i in imask]
                data['img_mask']=_int64_list_feature(imask)

                if use_font:
                    tgtw    =df.iloc[idx,4]    
                    # tgt
                    tmask=np.zeros(mask_dim)
                    # valid true format: processing in model
                    tmask[:,:tgtw]=1
                    tmask=tmask.flatten().tolist()
                    tmask=[int(i) for i in tmask]
                    data['tgt_mask']=_int64_list_feature(tmask)
            
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)  


#--------------------
# parsing util
#--------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#--------------------
# main
#--------------------
def store(cfg):
    '''
        stores tfrecords based on given config
    '''
    assert cfg.record_type in ['CRNN','ROBUSTSCANNER','ABINET'],"Wrong Record type:['CRNN','ROBUSTSCANNER','ABINET']"
    save_path        =   create_dir(cfg.save_path,"tfrecords")
    save_path        =   create_dir(save_path,cfg.iden)
    
    
    if cfg.record_type=="CRNN":
        pad_value=0
        create_mask=False
        start_end=0
    elif cfg.record_type=="ROBUSTSCANNER":
        c_start_end=len(cvocab)
        c_pad_value=len(cvocab)+1
        
        g_start_end=len(gvocab)
        g_pad_value=len(gvocab)+1
        
        create_mask=True
        pad_len    =80
        LOG_INFO(f"Grapheme Pad Value:{g_pad_value}")
        LOG_INFO(f"Unicode Pad Value:{c_pad_value}")
        
    else:
        raise NotImplementedError 

    #--------------------
    # src
    #--------------------    
    csv    =   os.path.join(cfg.data_path,"data.csv")
    img    =   os.path.join(cfg.data_path,"images")
    tgt    =   os.path.join(cfg.data_path,"targets")
    # process data_csv
    df=pd.read_csv(csv)
    # literal eval
    df.labels=df.labels.progress_apply(lambda x: literal_eval(x))
    # chars
    df["chars"]=df.labels.progress_apply(lambda x: [i for i in "".join(x)])
    # img paths
    df["img_path"]=df.filename.progress_apply(lambda x:os.path.join(img,x))
    # lengths
    df["lens"]=df.labels.progress_apply(lambda x:[len(x),len([i for i in "".join(x)])])
    df["lens"]=df.lens.progress_apply(lambda x:x if x[0]<=cfg.max_glen and x[1]<=cfg.max_clen else None)
    df.dropna(inplace=True)
    
    if cfg.record_type=="CRNN":
        # glabel clabel
        df["glabel"]=df.labels.progress_apply(lambda x: get_encoded_label(x,gvocab))
        df["clabel"]=df.chars.progress_apply(lambda x: get_encoded_label(x,cvocab))
        df["glabel"]=df.glabel.progress_apply(lambda x: pad_encoded_label(x,cfg.max_glen,pad_value))
        df["clabel"]=df.clabel.progress_apply(lambda x: pad_encoded_label(x,cfg.max_clen,pad_value))
    elif cfg.record_type=="ROBUSTSCANNER":
        df["glabel"]=df.labels.progress_apply(lambda x: get_encoded_label(x,gvocab[1:]))
        df["clabel"]=df.chars.progress_apply(lambda x: get_encoded_label(x,cvocab[1:]))
        # add tokens
        df["glabel"]=df.glabel.progress_apply(lambda x: [g_start_end]+x+[g_start_end])
        df["clabel"]=df.clabel.progress_apply(lambda x: [c_start_end]+x+[c_start_end])
        # pad
        df["glabel"]=df.glabel.progress_apply(lambda x: pad_encoded_label(x,pad_len,g_pad_value))
        df["clabel"]=df.clabel.progress_apply(lambda x: pad_encoded_label(x,pad_len,c_pad_value))

    if create_mask:
        if cfg.use_font:
            df=df[["img_path","glabel","clabel","image_mask","target_mask"]]
        else:
            df=df[["img_path","glabel","clabel","image_mask"]]
        # mask
        df["image_mask"]=df["image_mask"].progress_apply(lambda x:x if x > 0 else cfg.img_width)
        df["image_mask"]=df["image_mask"].progress_apply(lambda x: math.ceil((x/cfg.img_width)*(cfg.img_width//cfg.factor)))
        
        if cfg.use_font:
            df["target_mask"]=df["target_mask"].progress_apply(lambda x:x if x > 0 else cfg.img_width)
            df["target_mask"]=df["target_mask"].progress_apply(lambda x: math.ceil((x/cfg.img_width)*(cfg.img_width//cfg.factor)))

        mask_dim=(cfg.img_height//cfg.factor,cfg.img_width//cfg.factor)
    else:
        df=df[["img_path","glabel","clabel"]]
        mask_dim=None
        
    #--------------------
    # save
    #--------------------
    LOG_INFO(f"Creating TFRECORDS:{save_path}")
    for idx in tqdm(range(0,len(df),cfg.tf_size)):
        _df        =   df.iloc[idx:idx+cfg.tf_size]  
        rnum       =   idx//cfg.tf_size
        toTfrecord(_df,rnum,save_path,mask_dim,cfg.use_font)
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Script for Creating tfrecords")
    parser.add_argument("data_path",                                    help    ="Path of the processed data folder that contains images,targets and data.csv")
    parser.add_argument("save_path",                                    help    ="Path of the directory to save tfrecords")
    parser.add_argument("iden",                                         help    ="identifier of the dataset.")
    parser.add_argument("record_type",                                  help    ="specific record type to create. Availabe['CRNN','ROBUSTSCANNER','ABINET'] ")
    parser.add_argument("--img_height", required=False, default=64,     help    ="height for each grapheme: default=64")
    parser.add_argument("--img_width",  required=False, default=512,    help    ="width dimension of word images: default=512")
    parser.add_argument("--max_glen",   required=False, default=36,     help    ="maximum length of grapheme level data to keep: default=36")
    parser.add_argument("--max_clen",   required=False, default=62,     help    ="maximum length of unicode level data to keep: default=62")
    parser.add_argument("--tf_size",    required=False, default=1024,   help    ="number of data to keep in one record: default=1024")
    parser.add_argument("--factor",     required=False, default=32,     help    ="downscale factor for attention mask(used in robust scanner and abinet): default=32")
    parser.add_argument("--use_font",   required=False, default=False,  type=str2bool,help="Stores fontface images: default=False")
    
    
    
    args = parser.parse_args()

    class cfg:
        data_path   =   args.data_path
        save_path   =   args.save_path
        iden        =   args.iden
        record_type =   args.record_type
        img_width   =   int(args.img_width)
        img_height  =   int(args.img_height)
        max_glen    =   int(args.max_glen)
        max_clen    =   int(args.max_clen)
        tf_size     =   int(args.tf_size)
        factor      =   int(args.factor)
        use_font    =   args.use_font
        
    store(cfg)
    
    