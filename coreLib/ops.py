# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os 
import json
import cv2
import numpy as np
import pandas as pd 
import string


from collections import Counter
from glob import glob
from tqdm import tqdm
from .utils import LOG_INFO,create_dir
from .words import getRandomSyntheticData,cleanImage,padImage,get_data_frames,clean_non_found_graphemes,getTextImage


tqdm.pandas()
#--------------------
# GLOBALS
#--------------------
# symbols to avoid 
SYMBOLS = ['`','~','!','@','#','$','%',
           '^','&','*','(',')','_','-',
           '+','=','{','[','}','}','|',
           '\\',':',';','"',"'",'<',
           ',','>','.','?','/',
           '১','২','৩','৪','৫','৬','৭','৮','৯','০',
           '।']
SYMBOLS+=list(string.ascii_letters)
SYMBOLS+=[str(i) for i in range(10)]
#--------------------
# RESOURCES
#--------------------
LABEL_CSV   =   os.path.join(os.path.dirname(os.getcwd()),'resources','label.csv')
CLASS_CSV   =   os.path.join(os.path.dirname(os.getcwd()),'resources','classes.csv')
FONT_PATH   =   os.path.join(os.path.dirname(os.getcwd()),'resources','font.ttf')
DICT_CSV    =   os.path.join(os.path.dirname(os.getcwd()),'resources','words.csv')

#--------------------------------images2words------------------------------------------------------------
#--------------------
# helper functions
#--------------------

def extract_word_images_and_labels(img_path):
    '''
        extracts word images and labels from a given image
        args:
            img_path : path of the image
        returns:
            (images,labels)
            list of images and labels
    '''
    imgs=[]
    labels=[]
    # json_path
    json_path=img_path.replace("jpg","json")
    # read image
    data=cv2.imread(img_path,0)
    # label
    label_json = json.load(open(json_path,'r'))
    # get word idx
    for idx in range(len(label_json['shapes'])):
        # label
        label=str(label_json['shapes'][idx]['label'])
        # special charecter negation
        if not any(substring in label for substring in SYMBOLS):
            labels.append(label)
            # crop bbox
            xy=label_json['shapes'][idx]['points']
            # crop points
            x1 = int(np.round(xy[0][0]))
            y1 = int(np.round(xy[0][1]))
            x2 = int(np.round(xy[1][0]))
            y2 = int(np.round(xy[1][1]))
            # image
            img=data[y1:y2,x1:x2]
            imgs.append(img)
    return imgs,labels
#--------------------
# ops
#--------------------


def images2words(converted_path,save_path):
    '''
        creates the images based on labels
        args:
            converted_path   :  path of the converted folder
            save_path        :  path to save the dataset

        returns:
            * the dataset dataframe
            * location of the saved images
    '''
    img_idens=[]
    img_labels=[]
    i=0
    save_path=create_dir(save_path,'images2words')
    # get image paths
    img_paths=[img_path for img_path in glob(os.path.join(converted_path,"*.jpg"))]
    # iterate
    for img_path in tqdm(img_paths):
        # extract images and labels
        imgs,labels=extract_word_images_and_labels(img_path)
        if len(imgs)>0:
            for img,label in zip(imgs,labels):
                try:
                    # save path for the word
                    img_save_path=os.path.join(save_path,f"{i}.png")
                    # save
                    cv2.imwrite(img_save_path,img)
                    # append
                    img_idens.append(f"{i}.png")
                    img_labels.append(label)

                    i=i+1
                    
                except Exception as e: 
                    LOG_INFO(f"error in creating image:{img_path} label:{label},error:{e}",mcolor='red')
    # save to csv
    df=pd.DataFrame({"image_id":img_idens,"label":img_labels})
    return df,save_path
#--------------------------------cleandataset------------------------------------------------------------
#--------------------
# helper functions
#--------------------

def clean_red_flags(df,dict_words):
    '''
        removes the data row if the occurence of the label is only once
        args:
            df         :  the data frame that contains images2words image_id,label and graphemes
            dict_words :  list of dictionary words to check from
    '''
    # count dataframe
    count=Counter(df['label'].tolist())
    df_count = pd.DataFrame.from_dict(count, orient='index').reset_index()
    df_count = df_count.rename(columns={'index':'label', 0:'count'})
    
    # single instance
    df_count=df_count.loc[df_count["count"]==1]
    # check existance of problematic words
    red_flags=[]
    for label in tqdm(df_count.label):
        if label not in dict_words:
            red_flags.append(label)
    
    # clean dataframe
    df['label']=df['label'].progress_apply(lambda x: x if x not in red_flags else np.nan)
    df.dropna(inplace=True)
    return df
   
def sampleSingle(df):
    '''
        randomly samples a single image for each label
        args:
            df         :  the data frame that contains images2words image_id,label and graphemes
    '''
    dfs=[]
    # find labels
    unique_labels=set(df['label'].tolist())
    for label in tqdm(unique_labels):
        # label df
        _df=df.loc[df.label==label]
        # sample
        _df=_df.sample(n=1)
        #assert(len(_df)==1,"Houston we've got a problem")
        # append
        dfs.append(_df)
    df=pd.concat(dfs,ignore_index=True)
    return df


#--------------------
# ops
#--------------------
def cleanStyleTransferDataset(df):
    '''
        cleans the images2words dataset
        args:
            df   :   the data frame that contains images2words image_id,label
    '''
    # dictionary words
    dict_words=pd.read_csv(DICT_CSV).word.tolist()
    # get dfs
    df_root,df_vd,df_cd,df_grapheme=get_data_frames(class_map_csv=CLASS_CSV,
                                                    grapheme_labels_csv=LABEL_CSV)
    # non-found
    LOG_INFO("Cleaning non found graphemes")
    df=clean_non_found_graphemes(df=df,
                                 df_grapheme=df_grapheme,
                                 df_root=df_root,
                                 df_vd=df_vd,
                                 df_cd=df_cd,
                                 data_column="label",
                                 relevant_columns=["image_id","label"])
    # red flags
    LOG_INFO("Cleaning red flags")
    df=clean_red_flags(df=df,
                       dict_words=dict_words)
    
    # sample single
    LOG_INFO("Sample single")
    df=sampleSingle(df=df)
    return df
#--------------------------------StyleTransferDataset------------------------------------------------------------
#--------------------
# helper functions
#--------------------
def save_style_transfer_dataset(images2words_path,
                dataset,
                save_path,
                label_df,
                raw_path,
                img_height,
                data_dim):
    '''
        saves images and targets 
        args:
            images2words_path       :    location of images folder that contains images 2 words data in png
            dataset                 :    dataframe that contains labels and grapheme
            save_path               :    location of the  folder to save images and targets
            label_df                :    contains the labeled data from bengalai grapheme dataset  
            raw_path                :    directory that contains the raw grapheme images
            img_height              :    height for each grapheme
            data_dim                :    dimension of word images 
            

    '''
    images_path =create_dir(save_path,'images')
    targets_path=create_dir(save_path,'targets')
    for iid,grapheme_list in tqdm(zip(dataset['image_id'],dataset['graphemes']),total=len(dataset)):
        # img
        img=getRandomSyntheticData(grapheme_list=grapheme_list,
                                    label_df=label_df,
                                    png_dir=raw_path,
                                    img_height=img_height,
                                    data_dim=data_dim)
        cv2.imwrite(os.path.join(images_path,iid),img)
        # tgt
        img=cv2.imread(os.path.join(images2words_path,iid),0)
        img=cleanImage(img=img,
                       img_height=img_height)

        img=padImage(img=img,
                     data_dim=data_dim)
        cv2.imwrite(os.path.join(targets_path,iid),img)

#--------------------
# ops
#--------------------
def createStyleTransferDataset(dataset,
                  img_height,
                  data_dim,
                  raw_path,
                  images2words_path,
                  save_path):
    '''
        creates the images and targets for style transfer training
        args:
            dataset             :    clean dataset dataframe that contains image_id,label and graphemes
            img_height          :    height for each grapheme
            data_dim            :    dimension of word images 
            images2words_path   :    location of images folder that contains images 2 words data in png
            raw_path            :    directory that contains the raw grapheme images
            save_path           :    location to save the data
            
            
    '''
    # label
    label_df=pd.read_csv(LABEL_CSV)
    # create structre
    save_path=create_dir(save_path,'data')
    # save
    LOG_INFO("Saving  Data")
    save_style_transfer_dataset(images2words_path=images2words_path,
                                dataset=dataset,
                                save_path=save_path,
                                label_df=label_df,    
                                raw_path=raw_path,
                                img_height=img_height,
                                data_dim=data_dim)

    