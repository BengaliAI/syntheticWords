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
from .dfcore import clean_non_found_graphemes
from .imgcore import createWordImage,getRandomSyntheticData
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
CLASS_CSV   =   os.path.join(os.getcwd(),'resources','classes.csv')
LABEL_CSV   =   os.path.join(os.getcwd(),'resources','label.csv')
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
    LOG_INFO(save_path)
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

def clean_red_flags(df):
    '''
        removes the data row if the occurence of the label is only once
        args:
            df         :  the data frame that contains images2words image_id,label and graphemes
    '''
    # dictionary words
    dict_words=pd.read_csv(DICT_CSV).word.unique().tolist()
    
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
def cleanRecogDataset(dataset):
    '''
        cleans the images2words dataset
        args:
            dataset          :   the data frame that contains images2words image_id,label
        returns:
            the filtered version of the handwritten dataset  
    '''
    # non-found
    LOG_INFO("Cleaning non found graphemes")
    dataset=clean_non_found_graphemes(  df=dataset,
                                        data_column="label",
                                        relevant_columns=["image_id","label"])
    # red flags
    LOG_INFO("Cleaning red flags")
    dataset=clean_red_flags(df=dataset)
    

    return dataset
    



#--------------------
# ops
#--------------------
def createRecogDataset( dataset,
                        img_height,
                        img_width,
                        raw_path,
                        images2words_path,
                        save_path,
                        num_samples):
    '''
        creates the images and targets for style transfer training
        args:
            dataset             :    the filtered version of the handwritten dataset
            img_height          :    height for each grapheme
            img_width           :    width of word images 
            images2words_path   :    location of images folder that contains images 2 words data in png
            raw_path            :    directory that contains the raw grapheme images
            save_path           :    location to save the data
            num_samples         :    the amount of synthetic data to generate per word (default:5)
            
            
    '''
    
    # label
    label_df    =   pd.read_csv(LABEL_CSV)
    # create structre
    save_path   =   create_dir(save_path,'data')
    test_path   =   create_dir(save_path,'test')
    train_path  =   create_dir(save_path,'train')
    
    LOG_INFO("Creating testing Data")
    IMAGE_ID    =   []
    LABEL       =   []
    GRAPHEME    =   []
    count=0
    
    for iid,grapheme_list,label in tqdm(zip(dataset['image_id'],
                                            dataset['graphemes'],
                                            dataset['label']),
                                            total=len(dataset)):

        img=cv2.imread(os.path.join(images2words_path,iid),0)
        img=createWordImage(img=img,img_height=img_height,img_width=img_width)

        
        IMAGE_ID.append(f"test_{count}.png")             
        LABEL.append(label)
        GRAPHEME.append(grapheme_list)

        cv2.imwrite(os.path.join(test_path,f"test_{count}.png"),img)
        count+=1
    # save test images
    df=pd.DataFrame({"image":IMAGE_ID,
                     "label":LABEL,
                     "grapheme":GRAPHEME})
    df.to_csv(os.path.join(os.getcwd(),"resources","test.csv"),index=False)
    

    LOG_INFO("Creating training Data")
    IMAGE_ID    =   []
    LABEL       =   []
    GRAPHEME    =   []
    count=0
    # drop duplicates
    dataset=dataset.drop_duplicates(subset=['label'])
    
    for grapheme_list,label in tqdm(zip(dataset['graphemes'],
                                        dataset['label']),
                                        total=len(dataset)):    
        
        for _ in range(num_samples):
            img=getRandomSyntheticData( grapheme_list=grapheme_list,
                                        label_df=label_df,
                                        png_dir=raw_path,
                                        img_height=img_height,
                                        img_width=img_width)
            if img is not None:                            
                IMAGE_ID.append(f"train_{count}.png")             
                LABEL.append(label)
                GRAPHEME.append(grapheme_list)

                cv2.imwrite(os.path.join(train_path,f"train_{count}.png"),img)
                count+=1
        
    
    # save training images
    df=pd.DataFrame({"image":IMAGE_ID,
                     "label":LABEL,
                     "grapheme":GRAPHEME})
    df.to_csv(os.path.join(os.getcwd(),"resources","train.csv"),index=False)
    