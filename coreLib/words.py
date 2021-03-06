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
#--------------------------------------------------------------------------------------------
def stripPads(arr,
              val):
    '''
        strip specific value
        args:
            arr :   the numpy array (2d)
            val :   the value to strip
        returns:
            the clean array
    '''
    # x-axis
    arr=arr[~np.all(arr == val, axis=1)]
    # y-axis
    arr=arr[:, ~np.all(arr == val, axis=0)]
    return arr
#--------------------------------------------------------------------------------------------
def cleanImage(img,
               img_height):
    '''
        cleans and resizes the image after stripping
        args:
            img         :   numpy array grayscale image
            img_height  :   height for each grapheme
        returns:
            resized clean image
    '''
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # strip
    img=stripPads(arr=img,
                  val=255)
    # get shape
    h,w=img.shape
    _w=int(img_height*(w/h))
    # resize to char dim
    img=cv2.resize(img,(_w,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img
#--------------------------------------------------------------------------------------------
def getGraphemeImg(img_id,
                   png_dir,
                   img_height):
    '''
        reads and cleans a grapheme image 
        args:
            img_id      :  id of the png file
            png_dir     :  directory that contains the raw images
            img_height  :   height for each grapheme
        returns:
            binary grapheme image
    '''
    img_path  = os.path.join(png_dir,f"{img_id}.png")
    img       = cv2.imread(img_path,0)  
    img       = cleanImage(img=img,
                           img_height=img_height)
    return img
#--------------------------------------------------------------------------------------------
def padImage(img,
            data_dim):
    '''
        pads an image and resizes to (data_dim,data_dim)
        args:
            img         :   numpy array grayscale image
            data_dim    :   dimension of word images 
        returns:
            resized clean image
    '''
    # pad up-down
    h,w  = img.shape
    d_top=(data_dim-h)//2
    d_bot=data_dim-h-d_top
    top_pad=np.ones((d_top,w))*255
    bot_pad=np.ones((d_bot,w))*255
    # concat
    img=np.concatenate([top_pad,img,bot_pad],axis=0)
    img=img.astype('uint8')
    # resize
    img=cv2.resize(img,(data_dim,data_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img
#--------------------------------------------------------------------------------------------
def getRandomSyntheticData(grapheme_list,
                          label_df,
                          png_dir,
                          img_height,
                          data_dim):
    '''
        creates a synthetic data for a given list of graphemes
        args:
            grapheme_list  :  list of graphemes that MUST exist in label_csv
            label_df       :  contains the labeled data from bengalai grapheme dataset  
            png_dir        :  directory that contains the raw images
            img_height     :   height for each grapheme
            data_dim       :   dimension of word images 
            
        returns:
            an image of the lexicon that can be built from the given list
        
        **in the case of a non-found grapheme : 
            returns None for the image and the grapheme
    '''
    imgs=[]
    # iterate over the list
    for grapheme in grapheme_list:
        
        # get corresponding image ids for that grapheme
        grapheme_df=  label_df.loc[label_df.grapheme==grapheme]
        img_ids    =  grapheme_df.image_id.tolist()
        if len(img_ids)>0:
            # select a random one
            img_id     =  random.choice(img_ids)
            # get image
            img        =  getGraphemeImg(   img_id=img_id,
                                            png_dir=png_dir,
                                            img_height=img_height)
            imgs.append(img)
        else:
            return None

    if len(imgs)==0:
        return None
    # corner case
    if len(imgs)==1:
        img=imgs[0]
        img=padImage(img=img,
                    data_dim=data_dim)
        return img
    else:
        # combine
        img=np.concatenate(imgs,axis=1)
        # pad 
        img=padImage(img=img,
                    data_dim=data_dim)
        return img
#--------------------------------------------------------------------------------------------
def pad_text_image(img,
            data_dim):
    '''
        pads an image and resizes to (data_dim,data_dim)
        args:
            img         :   numpy array grayscale image
            data_dim    :   dimension of word images 
        returns:
            resized clean image
    '''
    # pad up-down
    h,w  = img.shape
    d_top=(data_dim-h)//2
    d_bot=data_dim-h-d_top
    top_pad=np.ones((d_top,w))*255
    bot_pad=np.ones((d_bot,w))*255
    # concat
    img=np.concatenate([top_pad,img,bot_pad],axis=0)
    # pad left-right
    h,w  = img.shape
    d_left=(data_dim-w)//2
    d_right=data_dim-w-d_left
    left_pad =np.ones((h,d_left))*255
    right_pad=np.ones((h,d_right))*255
    # concat
    img=np.concatenate([left_pad,img,right_pad],axis=1)
    # resize
    img=img.astype('uint8')    
    img=cv2.resize(img,(data_dim,data_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img
#--------------------------------------------------------------------------------------------
def getTextImage(text,_font,data_dim):
    '''
        create image from text
        args:
            text    :   the text to cleate the image of
            _font   :   the specific sized font to load
            data_dim:   the final dimension we need the image to be
        returns:
            the written text image
    '''

    WIDTH,HEIGHT=1024,1024
    # RGB image
    img = Image.new('RGB', (WIDTH,HEIGHT))
    # draw object
    draw = ImageDraw.Draw(img)
    # text height width
    w, h = draw.textsize(text, font=_font) 
    # drawing in the center
    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 2), text, font=_font)
    # grayscale
    img=img.convert('L')
    # array
    img=np.array(img)
    # iversion
    img=255-img
    # strip pads
    img=stripPads(img,255)
    # resize-image
    h,w=img.shape
    factor=1+(w//data_dim)
    if w>data_dim:
        # resize
        img=cv2.resize(img,(w//factor,h//factor),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    img=pad_text_image(img,data_dim)
    return img
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
#--------------------------------------------------------------------------------------------
