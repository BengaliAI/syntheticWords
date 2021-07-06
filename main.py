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

from coreLib.utils import LOG_INFO, get_encoded_label,get_sorted_vocab,pad_encoded_label
from coreLib.dataset import DataSet
from coreLib.bangla_writing import processData_BANGLA_WRITING
from coreLib.boise_state import processData_BOISE_STATE
from coreLib.bn_htr import processData_BN_HTR
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
    tf_size     =   1024
    # dataset object
    ds=DataSet(data_dir=data_path,save_path=save_path)
    
    #-----------------------------
    # bangla writing data
    #-----------------------------
    LOG_INFO("bangla writing data",mcolor="cyan")
    processData_BANGLA_WRITING(ds,dim=(img_height,img_width))
    #-----------------------------
    # boise state data
    #-----------------------------
    LOG_INFO("boise state data",mcolor="cyan")
    processData_BOISE_STATE(ds,dim=(img_height,img_width))
    #-----------------------------
    # bn htr data
    #-----------------------------
    LOG_INFO("bn htr data",mcolor="cyan")
    processData_BN_HTR(ds,dim=(img_height,img_width))
    #-----------------------------
    # process labels
    #-----------------------------
    # combine 
    ## bw split
    df_bw         =   pd.read_csv(ds.bangla_writing_csv)

    df_train      =   df_bw.loc[df_bw.src=="train"]
    df_train      =   df_train[["filename","word","graphemes","unicodes"]]
    
    df_eval       =   df_bw.loc[df_bw.src=="eval"]
    df_eval       =   df_eval[["filename","word","graphemes","unicodes"]]
    


    df_bh         =   pd.read_csv(ds.bn_htr_csv)
    df_bs         =   pd.read_csv(ds.boise_state_csv)
    df            =   pd.concat([df_train,df_eval,df_bh,df_bs],ignore_index=True)
    
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


    # df indexs:df_train,df_eval,df_bh,df_bs
    train_idx_start=0
    train_idx_end  =len(df_train)

    eval_idx_start =train_idx_end
    eval_idx_end   =train_idx_end+len(df_eval)

    bh_idx_start   =eval_idx_end
    bh_idx_end     =eval_idx_end+len(df_bh)

    bs_idx_start   =bh_idx_end
    bs_idx_end     =len(df)
    


    df_train      =   df.iloc[train_idx_start:train_idx_end]
    df_eval       =   df.iloc[eval_idx_start:eval_idx_end]
    df_bh         =   df.iloc[bh_idx_start:bh_idx_end]
    df_bs         =   df.iloc[bs_idx_start:bs_idx_end]

    
    
    # process
    df=df.drop_duplicates(subset=['word'])
    # error for none type
    df.dropna(inplace=True)
    # cleanup
    df.graphemes    =   df.graphemes.progress_apply(lambda x: x if set(x)<=set(ds.known_graphemes) else None)
    df.dropna(inplace=True)
    
    class word:
        data=df[["graphemes","clabel","glabel","word"]]
    
    ds.word=word
     
    
    LOG_INFO("Synthetic data",mcolor="cyan")
    # create synthetic data
    df_synth=createWords(ds,num_samples,dim=(img_height,img_width))
    
    # construct files
    df_train["img_path"]=df_train.filename.progress_apply(lambda x:os.path.join(ds.bangla_writing_path,x))
    df_eval["img_path"] =df_eval.filename.progress_apply(lambda x:os.path.join(ds.bangla_writing_path,x))
    df_bh["img_path"]=df_bh.filename.progress_apply(lambda x:os.path.join(ds.bn_htr_path,x))
    df_bs["img_path"]=df_bs.filename.progress_apply(lambda x:os.path.join(ds.boise_state_path,x))
    df_synth["img_path"]=df_synth.filename.progress_apply(lambda x:os.path.join(ds.synthetic_path,x))

    # saving
    df_train.to_csv(ds.bangla_writing_train_csv,index=False)
    df_eval.to_csv(ds.bangla_writing_eval_csv,index=False)
    df_bh.to_csv(ds.bn_htr_csv,index=False)
    df_bs.to_csv(ds.boise_state_csv,index=False)
    df_synth.to_csv(ds.synth_csv,index=False)   

    # format
    columns     =   ["img_path","clabel","glabel"]
    df_train    =   df_train[columns]
    df_eval     =   df_eval[columns]
    df_bh       =   df_bh[columns]
    df_bs       =   df_bs[columns]
    df_synth    =   df_synth[columns]   
    
    
    # create tfrecords
    df2rec(df_train,ds.tfrecords.train_bw,tf_size)
    df2rec(df_eval,ds.tfrecords.eval_bw,tf_size)
    
    df2rec(df_bs,ds.tfrecords.boise_state,tf_size)
    df2rec(df_bh,ds.tfrecords.bn_htr,tf_size)
    ## synthetic
    df2rec(df_synth,ds.tfrecords.synthetic,tf_size)
    
    # config 
    config={'img_height':img_height,
            'img_width':img_width,   
            'nb_channels':3,
            'nb_classes_char':len(cvocab),
            'nb_classes_grapheme':len(gvocab),
            'max_clabel_len':len(df_bs.iloc[0,1]),
            'max_glabel_len':len(df_bs.iloc[0,2]),
            'cvocab':cvocab,
            'gvocab':gvocab,
            'unique_words':len(ds.word.data)
            }
    with open(ds.config_json, 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=4,ensure_ascii=False)
    
               

#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Recognizer Training Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the source data folder ")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    parser.add_argument("--img_height",required=False,default=32,help ="height for each grapheme: default=32")
    parser.add_argument("--img_width",required=False,default=128,help ="width dimension of word images: default=128")
    parser.add_argument("--num_samples",required=False,default=100,help ="number of samples to create per word: default=100")
    args = parser.parse_args()
    main(args)
    
    