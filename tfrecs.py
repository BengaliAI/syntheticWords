#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import argparse
import os
import random
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd 
from ast import literal_eval
import json
from glob import glob
from tqdm import tqdm
from coreLib.utils import LOG_INFO,create_dir
tqdm.pandas()

#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_list_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#---------------------------------------------------------------
class Processor(object):
    def __init__(self,
                data_path,
                save_path,
                inp_len,
                data_size=20000):
        '''
            initializes the class
            args:
                data_path   =   location of raw data folder which contains eval and train folder
                save_path   =   location to save outputs (tfrecords,config.json)
                data_size   =   the size of tfrecords
                inp_len     =   CTC Decoder input time steps(if no step is ignored this is == img_width) 
        '''
        # public attributes
        self.data_path  =   data_path
        self.save_path  =   save_path
        self.data_size  =   data_size
        self.inp_len    =   inp_len 
        # private attributes
        self.__train_path   =   os.path.join(self.data_path,'train')
        self.__test_path    =   os.path.join(self.data_path,'test')
        
        # output paths
        self.__tfrec_path   =   create_dir(self.save_path,'tfrecords')
        self.__tfrec_train  =   create_dir(self.__tfrec_path,'train')
        self.__tfrec_test   =   create_dir(self.__tfrec_path,'test')
        
        self.__config_json  =   os.path.join(os.getcwd(),"resources",'config.json')
        # initialize dataframes
        LOG_INFO("create encoded labels")
        self.__createEncodedLabels()
    
    def __createEncodedLabels(self):
        '''
            initializes train and test dataframes
        '''
        train_df=pd.read_csv(os.path.join(os.getcwd(),"resources",'train.csv'))
        test_df=pd.read_csv(os.path.join(os.getcwd(),"resources",'test.csv'))
        # shuffle randomly
        train_df=train_df.sample(frac=1)
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
        # read as list
        train_df.grapheme=train_df.grapheme.progress_apply(lambda x: literal_eval(x))
        test_df.grapheme=test_df.grapheme.progress_apply(lambda x: literal_eval(x))
        # vocab
        vocab=[]
        for grapmeme_list in tqdm(train_df.grapheme):
            vocab+=grapmeme_list
        for grapmeme_list in tqdm(test_df.grapheme):
            vocab+=grapmeme_list
        vocab=sorted(list(set(vocab)))

        # local function for encoded label
        def get_encoded_label(grapheme_list):
            '''
                creates encoded label for images (multihot encoding)
            '''
            encoded=[]
            for grapheme in grapheme_list:
                encoded.append(vocab.index(grapheme))
            return encoded

        # encoded labels
        train_df["encoded"] =   train_df.grapheme.progress_apply(lambda x: get_encoded_label(x))
        test_df["encoded"]  =   test_df.grapheme.progress_apply(lambda x: get_encoded_label(x))
        #length
        train_df["label_length"]=train_df.grapheme.progress_apply(lambda x: len(x))
        test_df["label_length"]=test_df.grapheme.progress_apply(lambda x: len(x))
        max_label_len=max(train_df.label_length.max(),test_df.label_length.max())
        # set public attributes
        self.vocab              =   vocab
        self.train_df           =   train_df
        self.test_df            =   test_df
        self.max_label_len      =   int(max_label_len)
        self.pad_value          =   len(vocab)    


    def __toTfrecord(self):
        '''
        Creates tfrecords from Provided Image Paths
        '''
        tfrecord_name=f'{self.__rnum}.tfrecord'
        tfrecord_path=os.path.join(self.__rec_path,tfrecord_name) 
        #LOG_INFO(tfrecord_path)

        with tf.io.TFRecordWriter(tfrecord_path) as writer:    
            
            for _,row in self.__df.iterrows():
                img_path=os.path.join(self.__mode_path,row["image"]) 
                # img
                with(open(img_path,'rb')) as fid:
                    image_png_bytes=fid.read()
                # label
                label=row["encoded"]
                # label-length
                label_length=int(row["label_length"])

                for _ in range(self.max_label_len-len(label)):
                    label.append(self.pad_value)
                # feature desc
                data ={ 'image':_bytes_feature(image_png_bytes),
                        'label':_int64_list_feature(label),
                        'input_length':_int64_feature(self.inp_len),
                        'label_length':_int64_feature(label_length)
                }
                
                features=tf.train.Features(feature=data)
                example= tf.train.Example(features=features)
                serialized=example.SerializeToString()
                writer.write(serialized)  
            
    def __createRecs(self):
        '''
            tf record wrapper
        '''
        for idx in tqdm(range(0,len(self.df),self.data_size)):
            self.__df         =   self.df.iloc[idx:idx+self.data_size]  
            self.__rnum       =   idx//self.data_size
            self.__toTfrecord()

    def process(self):
        '''
            routine to create output
        '''
        # config.json
        ## format labels
        labels={}
        for idx,v in enumerate(self.vocab):
            labels[v]=idx


        _config={'img_height':IMG_HEIGHT,
                'img_width':IMG_WIDTH,   
                'nb_channels':NB_CHANNELS,
                'nb_classes':len(self.vocab),
                'pad_value':self.pad_value,
                'max_label_len':self.max_label_len,
                'nb_train_data':len(self.train_df),
                'nb_eval_data':len(self.test_df),
                'labels':labels
                }
        with open(self.__config_json, 'w') as fp:
            json.dump(_config, fp,sort_keys=True, indent=4,ensure_ascii=False)
        
        # create tf recs
        ## train
        self.df         =self.train_df
        self.__rec_path =self.__tfrec_train
        self.__mode_path=self.__train_path
        self.__createRecs()
        ## test 
        self.df         =self.test_df
        self.__rec_path =self.__tfrec_test
        self.__mode_path=self.__test_path
        self.__createRecs()
        
        


#---------------------------------------------------------------
def main(args):
    '''
        preprocesses data for training
        args:
            data_path   =   the location of folder that contains eval and train 
            save_path   =   path to save the tfrecords
            
    '''
    data_path   =   args.data_path
    save_path   =   args.save_path
    inp_len     =   int(args.input_length)
    processor_obj=Processor(data_path,save_path,inp_len)
    processor_obj.process()

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Tfrecords data for backbone-attention-ctc ")
    parser.add_argument("data_path", help="Path of the data folder that contains Test and Train")
    parser.add_argument("save_path", help="Path to save the tfrecords")
    parser.add_argument("--img_height",required=False,default=32,help ="height for each grapheme: default=32")
    parser.add_argument("--img_width",required=False,default=128,help ="width dimension of word images: default=128")
    parser.add_argument("--input_length",required=False,default=32,help="CTC Decoder input time steps(if no step is ignored this is == img_width) :default=32")
    parser.add_argument("--num_channels",required=False,default=3,help="number of channels in the images :default=3")
    
    args = parser.parse_args()
    # set globals
    IMG_HEIGHT  =   int(args.img_height)
    IMG_WIDTH   =   int(args.img_width)
    NB_CHANNELS =   int(args.num_channels)
    
    main(args)

    