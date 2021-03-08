# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd
import random
import cv2
import numpy as np
from tqdm import tqdm

tqdm.pandas()

#--------------------
# helpers
#--------------------

#--------------------------------------------------------------------------------------------
def word2grapheme(word,
                  df_root,
                  df_vd,
                  df_cd):
    '''
        creates a grapheme list for a given word
        args:
            word             :     the word to find grapheme for
            df_root          :     dataframe for grapheme roots
            df_vd            :     dataframe for vowel_diacritic 
            df_cd            :     dataframe for consonant_diacritic
            
        returns:
            list of graphemes
    '''
    graphemes = []
    grapheme = ''
    i = 0
    # iterate over the word
    while i < len(word):    
        grapheme+=(word[i])
        # pass for '্' 
        if word[i] in ['\u200d','্']:
            pass 
        # special case for 'ঁ'
        elif  word[i] == 'ঁ':
            graphemes.append(grapheme)
            grapheme = ''
        # if char in root    
        elif word[i] in df_root.values:
            if i+1 ==len(word):
                graphemes.append(grapheme)
            elif word[i+1] not in ['্', '\u200d', 'ঁ'] + list(df_vd.values):
                graphemes.append(grapheme)
                grapheme = ''
        # if char in vd 
        elif word[i] in df_vd.values:
            if i+1 ==len(word):
                graphemes.append(grapheme)
            elif word[i+1] != 'ঁ':
                graphemes.append(grapheme)
                grapheme = ''                

        
        
        i = i+1
    
    # filter  preceding '্' and cds+vds 
    #-----------> this is an aditional cleanup
    for grapheme in graphemes:
        if grapheme in list(df_cd.values)+ list(df_vd.values):
            graphemes.remove(grapheme)
    
    
        
    return graphemes



#--------------------
# dataframe ops
#--------------------

#--------------------------------------------------------------------------------------------
def get_data_frames(class_map_csv,
                    grapheme_labels_csv):
    '''
        reads and creates dataframe for roots,consonant_diacritic,vowel_diacritic and graphemes
        args:
            class_map_csv        : path of classes.csv
            grapheme_labels_csv  : path of labels.csv
        returns:
            tuple(df_root,df_vd,df_cd,df_grapheme)
            df_root          :     dataframe for grapheme roots
            df_vd            :     dataframe for vowel_diacritic 
            df_cd            :     dataframe for consonant_diacritic
            df_grapheme      :     dataframe for graphemes
            
    '''
    # read class map
    df_map=pd.read_csv(class_map_csv)
    # get grapheme roots
    df_root = df_map.groupby('component_type').get_group('grapheme_root')
    df_root.index = df_root['label']
    df_root = df_root.drop(columns = ['label','component_type'])
    # get vowel_diacritic
    df_vd = df_map.groupby('component_type').get_group('vowel_diacritic')
    df_vd.index = df_vd['label']
    df_vd = df_vd.drop(columns = ['label','component_type'])
    # get consonant_diacritic
    df_cd = df_map.groupby('component_type').get_group('consonant_diacritic')
    df_cd.index = df_cd['label']
    df_cd = df_cd.drop(columns = ['label','component_type'])
    # get grapheme labels
    df_grapheme=pd.read_csv(grapheme_labels_csv)
    # filter columns
    df_grapheme=df_grapheme[['image_id','grapheme']]
    return df_root,df_vd,df_cd,df_grapheme

#--------------------------------------------------------------------------------------------
def clean_non_found_graphemes(df,
                              df_grapheme,
                              df_root,
                              df_vd,
                              df_cd,
                              data_column,
                              relevant_columns):
    '''
       cleans non found graphemes
       args:
           df               :     dataframe where graphemes are to be created 
           df_grapheme      :     dataframe for graphemes
           df_root          :     dataframe for grapheme roots
           df_vd            :     dataframe for vowel_diacritic 
           df_cd            :     dataframe for consonant_diacritic
           data_column      :     name of the data column that needs conversion 
           relevant_columns :     list of the relevent columns to keep  
            
           
    '''
    # get graphemes from labels
    df['graphemes']=df[data_column].progress_apply(lambda x: word2grapheme(x,df_root,df_vd,df_cd))
    # find unique graphemes
    unique_graphemes=[]
    for grapheme_list in tqdm(df.graphemes.tolist()):
        unique_graphemes+=grapheme_list
    unique_graphemes=list(set(unique_graphemes))
    # find non found graphemes
    non_found_graphemes=[]
    for grapheme in tqdm(unique_graphemes):
        subset_df_grapheme=df_grapheme.loc[df_grapheme.grapheme==grapheme]
        if len(subset_df_grapheme)==0:
            non_found_graphemes.append(grapheme)
    # eliminate not found graphemes
    ##--> if the grapheme list intersects with a non found grapheme set it to nan
    df["not_found"]=df.graphemes.progress_apply(lambda x: np.nan if len(set(non_found_graphemes) & set(x))>0 else x)
    df.dropna(inplace=True)
    relevant_columns.append("graphemes")
    df=df[relevant_columns]
    return df 

