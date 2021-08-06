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
from .utils import LOG_INFO, correctPadding,create_dir,stripPads
import PIL
import PIL.Image , PIL.ImageDraw , PIL.ImageFont 
tqdm.pandas()
#--------------------
# helpers
#--------------------
def createImgFromComps(df,comps,pad):
    '''
        creates a synthetic image from given comps
        args:
            df      :       dataframe holding : "filename","label","img_path"
            comps   :       list of graphemes
            pad     :       pad class:
                                no_pad_dim
                                single_pad_dim
                                double_pad_dim
                                top
                                bot
        returns:
            non-pad-corrected raw binary image
    '''
    # get img_paths
    img_paths=[]
    for comp in comps:
        cdf=df.loc[df.label==comp]
        cdf=cdf.sample(frac=1)
        if len(cdf)==1:
            img_paths.append(cdf.iloc[0,2])
        else:
            img_paths.append(cdf.iloc[random.randint(0,len(cdf)-1),2])
    
    # get images
    imgs=[cv2.imread(img_path,0) for img_path in img_paths]
    
    # alignment of component
    ## flags
    tp=False
    bp=False
    comp_heights=["" for _ in comps]
    for idx,comp in enumerate(comps):
        if any(te.strip() in comp for te in pad.top):
            comp_heights[idx]+="t"
            tp=True
        if any(be in comp for be in pad.bot):
            comp_heights[idx]+="b"
            bp=True

    # image construction based on height flags
    '''
    class pad:
        no_pad_dim      =(comp_dim,comp_dim)
        single_pad_dim  =(int(comp_dim+pad_height),int(comp_dim+pad_height))
        double_pad_dim  =(int(comp_dim+2*pad_height),int(comp_dim+2*pad_height))
        top             =top_exts
        bot             =bot_exts
        height          =pad_height  
    '''
    cimgs=[]
    for img,hf in zip(imgs,comp_heights):
        if hf=="":
            img=cv2.resize(img,pad.no_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if tp:
                h,w=img.shape
                top=np.ones((pad.height,w))*255
                img=np.concatenate([top,img],axis=0)
            if bp:
                h,w=img.shape
                bot=np.ones((pad.height,w))*255
                img=np.concatenate([img,bot],axis=0)
        elif hf=="t":
            img=cv2.resize(img,pad.single_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if bp:
                h,w=img.shape
                bot=np.ones((pad.height,w))*255
                img=np.concatenate([img,bot],axis=0)

        elif hf=="b":
            img=cv2.resize(img,pad.single_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            if tp:
                h,w=img.shape
                top=np.ones((pad.height,w))*255
                img=np.concatenate([top,img],axis=0)
        elif hf=="bt" or hf=="tb":
            img=cv2.resize(img,pad.double_pad_dim,fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        
        cimgs.append(img)

    img=np.concatenate(cimgs,axis=1)
    return img 


def createTgtFromComps(font,comps,min_dim):
    '''
        creates font-space target images
        args:
            font    :   the font to use
            comps   :   the list of graphemes
            min_dim :   the minimum squre dimension to assign to each component==comp_dim
        return:
            non-pad-corrected raw binary target
    '''
    # max dim
    max_dim=len(comps)*min_dim
    # draw text
    image = PIL.Image.new(mode='L', size=(max_dim,max_dim))
    draw = PIL.ImageDraw.Draw(image)
    draw.text(xy=(0, 0), text="".join(comps), fill=255, font=font)
    # reverse
    tgt=np.array(image)
    idx=np.where(tgt>0)
    y_min,y_max,x_min,x_max = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
    tgt=tgt[y_min:y_max,x_min:x_max]
    #tgt=stripPads(tgt,0)
    tgt=255-tgt
    return tgt    
    
def createRandomDictionary(valid_graphemes,dict_max_len,dict_min_len,num_samples):
    '''
        creates a randomized dictionary
        args:
            valid_graphemes :       list of graphemes that can be used to create a randomized dictionary 
            num_samples     :       number of data to be created if no dictionary is provided
            dict_max_len    :       the maximum length of data for randomized dictionary
            dict_min_len    :       the minimum length of data for randomized dictionary
        returns:
            a dictionary dataframe with "word" and "graphemes"
    '''
    word=[]
    graphemes=[]
    for _ in tqdm(range(num_samples)):
        len_word=random.randint(dict_min_len,dict_max_len)
        _graphemes=[]
        for _ in range(len_word):
            _graphemes.append(random.choice(valid_graphemes))
        graphemes.append(_graphemes)
        word.append("".join(_graphemes))
    df=pd.DataFrame({"word":word,"graphemes":graphemes})
    return df 

def saveDictionary(dictionary,
                   compdf, 
                   save,
                   img_dim, 
                   pad,
                   font,
                   comp_dim):
    '''
        saves a dictionary data with img tgt and data.csv with grapheme labels
        args:
            dictionary      :       the dictionary to create img and tgt
            compdf          :       the dataframe that holds handwritten grapheme data ("filename","label","img_path")
            save            :       save class:
                                        img
                                        tgt
                                        csv
            img_dim         :       (img_height,img_width) tuple for final word image
            pad             :       padding class:
                                        no_pad_dim
                                        single_pad_dim
                                        double_pad_dim
                                        top
                                        bot
            font            :       font to use
            comp_dim        :       component height base
    '''
    # dataframe vars
    filename=[]
    labels=[]
    imasks=[]
    tmasks=[]
    # loop
    for idx in tqdm(range(len(dictionary))):
        try:
            comps=dictionary.iloc[idx,1]
            # image
            img=createImgFromComps(df=compdf,
                                comps=comps,
                                pad=pad)
            # target
            tgt=createTgtFromComps(font=font,
                                comps=comps,
                                min_dim=comp_dim)

            # correct padding
            img,imask=correctPadding(img,img_dim,ptype="left")
            tgt,tmask=correctPadding(tgt,img_dim,ptype="left")
            # save
            fname=f"{idx}.png"
            cv2.imwrite(os.path.join(save.img,fname),img)
            cv2.imwrite(os.path.join(save.tgt,fname),tgt)
            filename.append(fname)
            labels.append(comps)
            imasks.append(imask)
            tmasks.append(tmask)
        except Exception as e:
            LOG_INFO(e)
    df=pd.DataFrame({"filename":filename,"labels":labels,"image_mask":imasks,"target_mask":tmasks})
    df.to_csv(os.path.join(save.csv),index=False)

def saveFontFacedDictionary(dictionary,
                            all_fonts, 
                            save,
                            img_dim, 
                            font,
                            comp_dim):
    '''
        saves a dictionary data with img tgt and data.csv with grapheme labels
        args:
            dictionary      :       the dictionary to create img and tgt
            all_fonts       :       path for all fonts 
            save            :       save class:
                                        img
                                        tgt
                                        csv
            img_dim         :       (img_height,img_width) tuple for final word image
            font            :       font to use for target
            comp_dim        :       component height base
    '''
    # dataframe vars
    filename=[]
    labels=[]
    imasks=[]
    tmasks=[]
    # loop
    for idx in tqdm(range(len(dictionary))):
        try:
            comps=dictionary.iloc[idx,1]
            
            # resolution based font
            image_font_path=random.choice(all_fonts)
            res=random.choice(["low","mid"])
            if res=="low":
                image_font_size=comp_dim//4
            elif res=="mid":
                image_font_size=comp_dim//2
            image_font=font=PIL.ImageFont.truetype(image_font_path, size=image_font_size)
            # image            
            img=createTgtFromComps(font=image_font,
                                   comps=comps,
                                   min_dim=image_font_size)

            # resize (heigh based)
            _h,_w=img.shape 
            _width= int(comp_dim* _w/_h) 
            img=cv2.resize(img,(_width,comp_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            
            # target
            tgt=createTgtFromComps(font=font,
                                comps=comps,
                                min_dim=comp_dim)

            # correct padding
            img,imask=correctPadding(img,img_dim,ptype="left")
            tgt,tmask=correctPadding(tgt,img_dim,ptype="left")
            # save
            fname=f"{idx}.png"
            cv2.imwrite(os.path.join(save.img,fname),img)
            cv2.imwrite(os.path.join(save.tgt,fname),tgt)
            filename.append(fname)
            labels.append(comps)
            imasks.append(imask)
            tmasks.append(tmask)
        except Exception as e:
            LOG_INFO(e)
    df=pd.DataFrame({"filename":filename,"labels":labels,"image_mask":imasks,"target_mask":tmasks})
    df.to_csv(os.path.join(save.csv),index=False)




#--------------------
# ops
#--------------------
def createWords(iden,
                df,
                img_dir,
                save_dir,
                font_path,
                img_dim,
                comp_dim,
                pad_height,
                top_exts,
                bot_exts,
                dictionary=None,
                valid_graphemes=None,
                num_samples=100000,
                dict_max_len=10,
                dict_min_len=1):
    '''
        creates: 
            * handwriten word image
            * fontspace target image
            * a dataframe/csv that holds grapheme level groundtruth
        args:
            iden        :       identifier of the dataset
            df          :       the dataframe that contains filename and label 
            img_dir     :       the directory that holds the images
            save_dir    :       the directory to save the outputs
            font_path   :       the path of the font to be used
            img_dim         :       (img_height,img_width) tuple for final word image
            comp_dim        :       min component height for each grapheme image
            pad_height      :       the fixed padding height for alignment
            top_exts        :       list of extensions where the top is to be padded    
            bot_exts        :       list of extensions where the bottom is to be padded
            
            dictionary      :       if a dictionary is to be used, then pass the dictionary. 
                                    The dictionary dataframe should contain "word" and "graphemes"
                                    If None is provided: 
                                        Random combinations of graphemes will be provided 
                                        and num_samples data will created at random (default:100000)
            num_samples     :       number of data to be created (default:100000)
            **THESE ARGS ONLY WORK FOR dictionary=None case** 
            valid_graphemes :       list of graphemes that can be used to create a randomized dictionary 
             
            dict_max_len    :       the maximum length of data for randomized dictionary
            dict_min_len    :       the minimum length of data for randomized dictionary
    '''
    #---------------
    # processing
    #---------------
    save_dir=create_dir(save_dir,iden)
    # create img_path in df
    df["img_path"]=df.filename.progress_apply(lambda x:os.path.join(img_dir,f"{x}.bmp")) 
    # save_paths
    class save:    
        img=create_dir(save_dir,"images")
        tgt=create_dir(save_dir,"targets")
        csv=os.path.join(save_dir,"data.csv")
    # font 
    font=PIL.ImageFont.truetype(font_path, size=comp_dim)
    # pad
    class pad:
        no_pad_dim      =(comp_dim,comp_dim)
        single_pad_dim  =(int(comp_dim+pad_height),int(comp_dim+pad_height))
        double_pad_dim  =(int(comp_dim+2*pad_height),int(comp_dim+2*pad_height))
        top             =top_exts
        bot             =bot_exts
        height          =pad_height   

    # handle dict
    if dictionary is None:
        dictionary=createRandomDictionary(valid_graphemes,dict_max_len,dict_min_len,num_samples)
    else:
        dictionary=dictionary.sample(frac=1)
        dictionary=dictionary.head(num_samples)
    # save data
    saveDictionary(dictionary=dictionary,
                   compdf=df,
                   save=save,
                   img_dim=img_dim,
                   pad=pad,
                   font=font,
                   comp_dim=comp_dim)


def createFontFacedWords(iden,
                        save_dir,
                        all_fonts,
                        font_path,
                        img_dim,
                        comp_dim,
                        valid_graphemes,
                        num_samples=100000,
                        dict_max_len=10,
                        dict_min_len=1):
    '''
        creates: 
            * handwriten word image
            * fontspace target image
            * a dataframe/csv that holds grapheme level groundtruth
        args:
            iden            :       identifier of the dataset
            save_dir        :       the directory to save the outputs
            all_fonts       :       all the available fonts
            font_path       :       the path of the font to be used for target 
            img_dim         :       (img_height,img_width) tuple for final word image
            comp_dim        :       min component height for each grapheme image
            num_samples     :       number of data to be created (default:100000)
            valid_graphemes :       list of graphemes that can be used to create a randomized dictionary  
            dict_max_len    :       the maximum length of data for randomized dictionary
            dict_min_len    :       the minimum length of data for randomized dictionary
    '''
    #---------------
    # processing
    #---------------
    save_dir=create_dir(save_dir,iden)
    # save_paths
    class save:    
        img=create_dir(save_dir,"images")
        tgt=create_dir(save_dir,"targets")
        csv=os.path.join(save_dir,"data.csv")
    # font 
    font=PIL.ImageFont.truetype(font_path, size=comp_dim)

    dictionary=createRandomDictionary(valid_graphemes,dict_max_len,dict_min_len,num_samples)
    
    # save data
    saveFontFacedDictionary(dictionary=dictionary,
                            all_fonts=all_fonts,
                            save=save,
                            img_dim=img_dim,
                            font=font,
                            comp_dim=comp_dim)