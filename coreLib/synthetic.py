# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import cv2
import numpy as np
import random
import pandas as pd 
from tqdm.auto import tqdm
from .utils import *
#--------------------
# helpers
#--------------------
def padImage(img,pad_loc,pad_dim):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w=img.shape
        # pad widths
        left_pad_width =(pad_dim-w)//2
        # print(left_pad_width)
        right_pad_width=pad_dim-w-left_pad_width
        # pads
        left_pad =np.ones((h,left_pad_width))*255
        right_pad=np.ones((h,right_pad_width))*255
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
    else:
        # shape
        h,w=img.shape
        # pad heights
        top_pad_height =(pad_dim-h)//2
        bot_pad_height=pad_dim-h-top_pad_height
        # pads
        top_pad =np.ones((top_pad_height,w))*255
        bot_pad=np.ones((bot_pad_height,w))*255
        # pad
        img =np.concatenate([top_pad,img,bot_pad],axis=0)
    return img.astype("uint8")    

#--------------------
# ops
#--------------------
def createWords(ds,num_samples,dim=(128,32)):
    '''
        creates handwriten word image
        args:
            ds          :       the dataset object
            num_samples :       number of samples per word
            dim         :       (img_width,img_height) tuple to resize to 

    '''
    synth_path=ds.synthetic_path
    
    mods=['ঁ', 'ং', 'ঃ']
    # initialize image count
    img_count=0
    # extract dimensions
    img_width,img_height=dim
    # dataframe
    df=ds.word.data

    # synthetic dataset containers
    img_paths=[]
    clabels=[]
    glabels=[]

    for idx in tqdm(range(len(df))):
        # extract values
        graphemes=df.iloc[idx,0]
        clabel   =df.iloc[idx,1]
        glabel   =df.iloc[idx,2]
        
        # reconfigure comps
        while graphemes[0] in mods:
            graphemes=graphemes[1:]
        
        # collect data frames for each grapheme
        comp_dfs=[ds.graphemes.df.loc[ds.graphemes.df.label==grapheme] for grapheme in graphemes]    
        
        
        # run for n number of samples
        for _ in range(num_samples):
            imgs=[]
            for c_df in comp_dfs:             
                # select a image file
                idx=random.randint(0,len(c_df)-1)
                img_path=os.path.join(ds.graphemes.dir,f"{c_df.iloc[idx,0]}.bmp") 
                # read image
                img=cv2.imread(img_path,0)
                # resize
                h,w=img.shape 
                width= int(img_height* w/h) 
                img=cv2.resize(img,(width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
                imgs.append(img)
            img=np.concatenate(imgs,axis=1)
            # check for pad
            h,w=img.shape
            if w > img_width:
                # for larger width
                h_new= int(img_width* h/w) 
                img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
                # pad
                img=padImage(img,pad_loc="tb",pad_dim=img_height) 
            elif w < img_width:
                # pad
                img=padImage(img,pad_loc="lr",pad_dim=img_width)

            # save the image
            img_path=os.path.join(synth_path,f"{img_count}.png")
            cv2.imwrite(img_path,img)
            img_count+=1
            # data variables
            img_paths.append(img_path)
            glabels.append(glabel)
            clabels.append(clabel)
    
    # dataframe
    df=pd.DataFrame({"img_path":img_paths,"clabel":clabels,"glabel":glabels})
    df=df.sample(frac=1)

    class synthetic:
        data=df

    ds.synthetic=synthetic
    return ds  


