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
from tqdm import tqdm
from .utils import correctPadding
tqdm.pandas()
#--------------------
# ops
#--------------------
def createWords(ds,num_samples,dim=(32,256)):
    '''
        creates handwriten word image
        args:
            ds          :       the dataset object
            num_samples :       number of samples per word
            dim         :       (img_height,img_width) tuple to resize to 

    '''
    (img_height,img_width)=dim 

    synth_path=ds.synthetic_path
    
    mods=['ঁ', 'ং', 'ঃ']
    # initialize image count
    img_count=0
    
    # dataframe
    df=ds.word.data

    # synthetic dataset containers
    img_paths=[]
    clabels=[]
    glabels=[]
    words=[]
    img_labels=[]

    for idx in tqdm(range(len(df))):
        # extract values
        graphemes=df.iloc[idx,0]
        clabel   =df.iloc[idx,1]
        glabel   =df.iloc[idx,2]
        word     =df.iloc[idx,3]

        try:
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
                    img[img>0]=255
                    # resize
                    h,w=img.shape 
                    width= int(img_height* w/h) 
                    img=cv2.resize(img,(width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
                    imgs.append(img)
                
                
                img=np.concatenate(imgs,axis=1)
                # resize (heigh based)
                h,w=img.shape 
                width= int(img_height* w/h) 
                img=cv2.resize(img,(width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
                # save
                img=correctPadding(img,dim)     
                # save the image
                img_path=os.path.join(synth_path,f"{img_count}.png")
                cv2.imwrite(img_path,img)
                
                
                
                
                # data variables
                img_paths.append(f"{img_count}.png")
                glabels.append(glabel)
                clabels.append(clabel)
                words.append(word)
                img_labels.append(graphemes)

                img_count+=1
        
        except Exception as e:
            print(e)

    # dataframe
    df=pd.DataFrame({"filename":img_paths,"word":words,"graphemes":img_labels,"clabel":clabels,"glabel":glabels})
    # unicodes
    df["unicodes"]  =   df.word.progress_apply(lambda x:[i for i in x])
    df=df[["filename","word","graphemes","unicodes","clabel","glabel"]]
    df.to_csv(ds.synth_csv,index=False)

    return df  


