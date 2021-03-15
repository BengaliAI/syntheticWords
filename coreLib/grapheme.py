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
import regex
tqdm.pandas()

#--------------------
# globals
#--------------------
CLASS_CSV   =   os.path.join(os.getcwd(),'resources','classes.csv')
#--------------------
# helpers
#--------------------

def get_data_frames(class_map_csv):
    '''
        reads and creates dataframe for roots,consonant_diacritic,vowel_diacritic and graphemes
        args:
            class_map_csv        : path of classes.csv
        returns:
            tuple(df_root,df_vd,df_cd)
            df_root          :     dataframe for grapheme roots
            df_vd            :     dataframe for vowel_diacritic 
            df_cd            :     dataframe for consonant_diacritic
            
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
    return df_root,df_vd,df_cd

#------------
# global ops
#-----------
df_root,df_vd,df_cd=get_data_frames(class_map_csv=CLASS_CSV)
vds=df_vd.component.tolist()[1:]
cds=df_cd.component.tolist()[1:]

mds=['়']
rts=df_root.component.tolist()

chs=[]
for root in rts:
    decomp=[ch for ch in root]
    if len(decomp)==1:
        chs.append(decomp[0])
nms=['১','২','৩','৪','৫','৬','৭','৮','৯','০']
#-----------
# helpers
#-----------
def cleanWord(word):
    '''
        cleans a word from any non benglai chars and numbers
        args:
            word   :   the word to clean
        return:
            a list of decomposed chars
    '''
    if word[0]=='়' or word[0]=='্':
        word=word[1:]
    for num in nms:
        if num in word:
            word=word.replace(num,"")
    decomp=regex.findall(r"[\p{Bengali}]",word)
    return decomp

def get_root_from_decomp(decomp):
    '''
        creates grapheme root based list 
    '''
    # mod correction
    for idx,d in enumerate(decomp):
        if d==mds[0]:
            decomp[idx-1]=decomp[idx-1]+mds[0]
    while mds[0] in decomp:
        decomp.remove(mds[0])
            
    # map roots
    connector= '্'
    if connector in decomp:
        c_idxs = [i for i, x in enumerate(decomp) if x == connector]
        # component wise index map    
        comps=[[cid-1,cid,cid+1] for cid in c_idxs ]
        # merge multi root
        r_decomp = []
        while len(comps)>0:
            first, *rest = comps
            first = set(first)

            lf = -1
            while len(first)>lf:
                lf = len(first)

                rest2 = []
                for r in rest:
                    if len(first.intersection(set(r)))>0:
                        first |= set(r)
                    else:
                        rest2.append(r)     
                rest = rest2

            r_decomp.append(sorted(list(first)))
            comps = rest
        # add    
        combs=[]
        for ridx in r_decomp:
            comb=''
            for i in ridx:
                comb+=decomp[i]
                combs.append(comb)
            for i in ridx:
                decomp[i]=comb
        # new root based decomp
        new_decomp=[]
        for i in range(len(decomp)-1):
            if decomp[i]!=decomp[i+1]:
                new_decomp.append(decomp[i])
        new_decomp.append(decomp[-1])
        return new_decomp
    else:
        return decomp

def get_graphemes_from_decomp(decomp):
    '''
        create graphemes from decomp
    '''
    graphemes=[]
    idxs=[]
    for idx,d in enumerate(decomp):
        if d not in vds+cds+mds:
            idxs.append(idx)
    idxs.append(len(decomp))
    for i in range(len(idxs)-1):
        sub=decomp[idxs[i]:idxs[i+1]]
        grapheme=''
        for s in sub:
            grapheme+=s
        graphemes.append(grapheme)
    return graphemes

#-----------
# ops
#-----------
def word2grapheme(word):
    '''
        creates grapheme list for a given word
    '''
    decomp=cleanWord(word)
    decomp=get_root_from_decomp(decomp)
    graphemes=get_graphemes_from_decomp(decomp)
    return graphemes