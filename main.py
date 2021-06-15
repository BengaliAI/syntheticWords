#!/usr/bin/python3
# -*-coding: utf-8 -
'''
    @author:  MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import argparse
import os 
import json

from coreLib.utils import *
from coreLib.dataset import DataSet
from coreLib.test_data import pages2words
from coreLib.synthetic import createWords
from coreLib.store import df2rec
#--------------------
# main
#--------------------
def main(args):
    '''
        * Creates a pages2words data
        * creates a dataset for Recognizer Training
    '''  
    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    num_samples =   int(args.num_samples)
    tf_size     =   20000
    # dataset object
    ds=DataSet(data_dir=data_path,save_path=save_path)
    # create pages 2 words
    ds=pages2words(ds,dim=(img_height,img_width))

    # create synthetic data
    #ds=createWords(ds,num_samples,dim=(img_width,img_height))
    
    
    
    # # create tfrecords
    # ## train
    # df2rec(ds.word.train,ds.tfrecords.train,tf_size)
    # ## test
    # df2rec(ds.word.test,ds.tfrecords.test,tf_size)
    # ## synthetic
    # df2rec(ds.synthetic.data,ds.tfrecords.synthetic,tf_size)
    
    # # config 
    # config={'img_height':img_height,
    #         'img_width':img_width,   
    #         'nb_channels':3,
    #         'nb_classes_char':len(ds.vocab.charecter),
    #         'nb_classes_grapheme':len(ds.vocab.grapheme),
    #         'max_clabel_len':len(ds.word.data.iloc[0,1]),
    #         'max_glabel_len':len(ds.word.data.iloc[0,2]),
    #         'nb_train':len(ds.word.train),
    #         'nb_test' :len(ds.word.test),
    #         'nb_synth':len(ds.synthetic.data),
    #         'cvocab':ds.vocab.charecter,
    #         'gvocab':ds.vocab.grapheme,
    #         'unique_words':len(ds.word.data)
    #         }
    # with open(ds.config_json, 'w') as fp:
    #     json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)
    
               
    
#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the data folder that contains converted and raw folder from ReadMe.md)")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=32,help ="height for each grapheme: default=32")
    parser.add_argument("--img_width",required=False,default=256,help ="width dimension of word images: default=256")
    parser.add_argument("--num_samples",required=False,default=200,help ="number of samples to create per word: default=200")
    args = parser.parse_args()
    main(args)
    
    