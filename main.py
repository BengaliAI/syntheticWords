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
import pandas as pd 

from tqdm import tqdm
from ast import literal_eval

from coreLib.utils import get_encoded_label,get_sorted_vocab,pad_encoded_label
from coreLib.dataset import DataSet
from coreLib.test_data import processTestData
from coreLib.train_data import processTrainData
from coreLib.synthetic import createWords
from coreLib.store import df2rec
tqdm.pandas()
#--------------------
# main
#--------------------
def main(args):

    data_path   =   args.data_path
    save_path   =   args.save_path
    img_height  =   int(args.img_height)
    img_width   =   int(args.img_width)
    num_samples =   int(args.num_samples)
    tf_size     =   20000
    # dataset object
    ds=DataSet(data_dir=data_path,save_path=save_path)
    
    #-----------------------------
    # test data
    #-----------------------------
    processTestData(ds,dim=(img_height,img_width))
    #-----------------------------
    # train data
    #-----------------------------
    processTrainData(ds,dim=(img_height,img_width))
    
    #-----------------------------
    # process labels
    #-----------------------------
    # combine
    df_test         =   pd.read_csv(ds.test_csv)
    df_train        =   pd.read_csv(ds.train_csv)
    df              =   pd.concat([df_test,df_train],ignore_index=True)
    # eval
    df.graphemes    =   df.graphemes.progress_apply(lambda x: literal_eval(x))
    df.unicodes     =   df.unicodes.progress_apply(lambda x: literal_eval(x))

    # char vocab
    symbol_lists    =    df.unicodes.tolist()
    cvocab          =    get_sorted_vocab(symbol_lists)
    max_len         =    max([len(l) for l in symbol_lists])
    df["clabel"]    =    df.unicodes.progress_apply(lambda x: get_encoded_label(x,cvocab))
    df.clabel       =    df.clabel.progress_apply(lambda x: pad_encoded_label(x,max_len,0))

    # grapheme vocab
    symbol_lists    =    df.graphemes.tolist()
    gvocab          =    get_sorted_vocab(symbol_lists)
    max_len         =    max([len(l) for l in symbol_lists])
    df["glabel"]    =    df.graphemes.progress_apply(lambda x: get_encoded_label(x,gvocab))
    df.glabel       =    df.glabel.progress_apply(lambda x: pad_encoded_label(x,max_len,0))    

    df_test         =   df.loc[:len(df_test)]
    df_train        =   df.loc[len(df_test):]

    df_test.to_csv(ds.test_csv,index=False)
    df_train.to_csv(ds.train_csv,index=False)

    class word:
        # synthetic words dataframe
        df=df_test.drop_duplicates(subset=['word'])
        data=df[["graphemes","clabel","glabel","word"]]
    
    ds.word=word
     
    

    # create synthetic data
    df_synth=createWords(ds,num_samples,dim=(img_height,img_width))
    
    # construct files
    df_test["img_path"]=df_test.filename.progress_apply(lambda x:os.path.join(ds.test_path,x))
    df_train["img_path"]=df_train.filename.progress_apply(lambda x:os.path.join(ds.train_path,x))
    #df_synth["img_path"]=df_synth.filename.progress_apply(lambda x:os.path.join(ds.save_path,x))

    # format
    columns     =   ["img_path","clabel","glabel"]
    df_test     =   df_test[columns]
    df_train    =   df_train[columns]
    df_synth    =   df_synth[columns]   
    
    
    # create tfrecords
    ## train
    df2rec(df_train,ds.tfrecords.train,tf_size)
    ## test
    df2rec(df_test,ds.tfrecords.test,tf_size)
    ## synthetic
    df2rec(df_synth,ds.tfrecords.synthetic,tf_size)
    
    # config 
    config={'img_height':img_height,
            'img_width':img_width,   
            'nb_channels':3,
            'nb_classes_char':len(cvocab),
            'nb_classes_grapheme':len(gvocab),
            'max_clabel_len':len(df_test.iloc[0,1]),
            'max_glabel_len':len(df_test.iloc[0,2]),
            'nb_train':len(df_train),
            'nb_test' :len(df_test),
            'nb_synth':len(df_synth),
            'cvocab':cvocab,
            'gvocab':gvocab
            }
    with open(ds.config_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)
    
               
    
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
    
    