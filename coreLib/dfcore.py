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
# parser
#--------------------
from .graphemeParser import GraphemeParser
#--------------------
# global
#--------------------

CLASS_CSV   =   os.path.join(os.getcwd(),'resources','classes.csv')
LABEL_CSV   =   os.path.join(os.getcwd(),'resources','label.csv')

GP          =   GraphemeParser(CLASS_CSV)

# get grapheme labels
df_grapheme=pd.read_csv(LABEL_CSV)
# filter columns
df_grapheme=df_grapheme[['image_id','grapheme']]


#--------------------------------------------------------------------------------------------
def clean_non_found_graphemes(df,
                              data_column,
                              relevant_columns):
    '''
       cleans non found graphemes
       args:
            df                      :     dataframe where graphemes are to be created 
            data_column             :     name of the data column that needs conversion 
            relevant_columns        :     list of the relevent columns to keep  
            
           
    '''
    # get graphemes from labels
    df['graphemes']=df[data_column].progress_apply(lambda x: GP.word2grapheme(x))
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

