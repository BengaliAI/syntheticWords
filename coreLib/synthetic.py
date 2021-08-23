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


def createTgtFromComps(font,comps):
    '''
        creates font-space target images
        args:
            font    :   the font to use
            comps   :   the list of graphemes
        return:
            non-pad-corrected raw binary target
    '''
    
    # draw text
    image = PIL.Image.new(mode='L', size=font.getsize("".join(comps)))
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
    



#--------------------
# ops
#--------------------
def createSyntheticData(iden,
                        df,
                        img_dir,
                        save_dir,
                        fonts,
                        img_dim,
                        comp_dim,
                        pad_height,
                        top_exts,
                        bot_exts,
                        dictionary,
                        sample_per_word=100):
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
            fonts       :       the path of the fonts to be used
            img_dim         :       (img_height,img_width) tuple for final word image
            comp_dim        :       min component height for each grapheme image
            pad_height      :       the fixed padding height for alignment
            top_exts        :       list of extensions where the top is to be padded    
            bot_exts        :       list of extensions where the bottom is to be padded
            
            dictionary      :       if a dictionary is to be used, then pass the dictionary. 
                                    The dictionary dataframe should contain "word" and "graphemes"
            sample_per_word :       number of samples per word
                                
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
        csv=os.path.join(save_dir,"data.csv")
    # pad
    class pad:
        no_pad_dim      =(comp_dim,comp_dim)
        single_pad_dim  =(int(comp_dim+pad_height),int(comp_dim+pad_height))
        double_pad_dim  =(int(comp_dim+2*pad_height),int(comp_dim+2*pad_height))
        top             =top_exts
        bot             =bot_exts
        height          =pad_height   

    
    dictionary=dictionary.sample(frac=1)
        
    # save data
    # dataframe vars
    filename=[]
    labels=[]
    imasks=[]
    # loop
    for idx in tqdm(range(len(dictionary))):
        try:
            comps=dictionary.iloc[idx,1]
            for font_path in fonts:
                font=PIL.ImageFont.truetype(font_path,comp_dim)

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
    df=pd.DataFrame({"filename":filename,"labels":labels,"image_mask":imasks})
    df.to_csv(os.path.join(save.csv),index=False)
