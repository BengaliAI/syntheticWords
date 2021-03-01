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
from .words import getRandomSyntheticData,cleanImage,padImage


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
LABEL_CSV   =   os.path.join(os.getcwd(),'resources','label.csv')
CLASS_CSV   =   os.path.join(os.getcwd(),'resources','classes.csv')
FONT_PATH   =   os.path.join(os.getcwd(),'resources','font.ttf')
DICT_CSV    =   os.path.join(os.getcwd(),'resources','words.csv')

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

def clean_non_found_graphemes(df,
                              df_grapheme,
                              df_root,
                              df_vd,
                              df_cd):
    '''
       cleans non found graphemes
       args:
           df               :     dataframe for image and label in image2words 
           df_grapheme      :     dataframe for graphemes
           df_root          :     dataframe for grapheme roots
           df_vd            :     dataframe for vowel_diacritic 
           df_cd            :     dataframe for consonant_diacritic
            
           
    '''
    # get graphemes from labels
    df['graphemes']=df['label'].progress_apply(lambda x: word2grapheme(x,df_root,df_vd,df_cd))
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
    df=df[['image_id','label','graphemes']]
    return df 

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
   
#--------------------
# ops
#--------------------
def cleanDataset(df):
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
                                 df_cd=df_cd)
    # red flags
    LOG_INFO("Cleaning red flags")
    df=clean_red_flags(df=df,
                       dict_words=dict_words)
    return df
#--------------------------------images2words------------------------------------------------------------
#--------------------
# helper functions
#--------------------
def save_dataset(data_path,
                ds_df,
                save_path,
                label_df,
                raw_path,
                img_height,
                data_dim):
    '''
        saves images and targets 
        args:
            data_path      :    location of images folder that contains images 2 words data in png
            ds_df          :    dataframe that contains labels and grapheme
            save_path      :    location of the mode(test/train) folder
            label_df       :    contains the labeled data from bengalai grapheme dataset  
            raw_path       :    directory that contains the raw grapheme images
            img_height     :    height for each grapheme
            data_dim       :    dimension of word images 
            

    '''
    count=0
    images_path =create_dir(save_path,'images')
    targets_path=create_dir(save_path,'targets')
    for iid,grapheme_list in tqdm(zip(ds_df['image_id'],ds_df['graphemes']),total=len(ds_df)):
        # img
        img=getRandomSyntheticData(grapheme_list=grapheme_list,
                                    label_df=label_df,
                                    png_dir=raw_path,
                                    img_height=img_height,
                                    data_dim=data_dim)
        cv2.imwrite(os.path.join(images_path,f'{count}.png'),img)
        # tgt
        img=cv2.imread(os.path.join(data_path,iid),0)
        img=cleanImage(img=img,
                       img_height=img_height)

        img=padImage(img=img,
                     data_dim=data_dim)
        cv2.imwrite(os.path.join(targets_path,f'{count}.png'),img)

        count+=1
#--------------------
# ops
#--------------------
def createDataset(dataset,
                  img_height,
                  data_dim,
                  raw_path,
                  images2words_path,
                  save_path,
                  split=0.1):
    '''
        creates the images and targets for style transfer training
        args:
            dataset             :    clean dataset dataframe that contains image_id,label and graphemes
            img_height          :    height for each grapheme
            data_dim            :    dimension of word images 
            images2words_path   :    location of images folder that contains images 2 words data in png
            raw_path            :    directory that contains the raw grapheme images
            save_path           :    location to save the data
            split               :    float percent of eval split (default 0.1 i.e-10%)
            
            
    '''
    # label
    label_df=pd.read_csv(LABEL_CSV)
    # create structre
    save_path=create_dir(save_path,'data')
    # test train
    train_path=create_dir(save_path,'train')
    test_path =create_dir(save_path,'test')
    # split
    nb_train=int(len(dataset)*(1-split))
    train_df=dataset.head(nb_train)
    test_df =dataset.tail(len(dataset)-nb_train)
    # save
    LOG_INFO("Saving Training Data")
    save_dataset(data_path=images2words_path,
                ds_df=train_df,
                save_path=train_path,
                label_df=label_df,    
                raw_path=raw_path,
                img_height=img_height,
                data_dim=data_dim)

    LOG_INFO("Saving Testing Data")
    save_dataset(data_path=images2words_path,
                ds_df=test_df,
                save_path=test_path,
                label_df=label_df,    
                raw_path=raw_path,
                img_height=img_height,
                data_dim=data_dim)
