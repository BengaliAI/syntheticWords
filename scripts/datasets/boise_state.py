# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')
#----------------------
# imports
#----------------------
import argparse
import os 
import pandas as pd 
import cv2
from coreLib.utils import *
from glob import glob
from tqdm.auto import tqdm
tqdm.pandas()
import random
random.seed(42)

def extract_info(_dir,coords,fmt):
    '''
        extracts information from boise-state annotations
    '''
    img_paths=[img_path for img_path in glob(os.path.join(_dir,f"*.{fmt}"))]
    liness=[]
    words=[]
    comps=[]
    chars=[]
    xmins=[]
    ymins=[]
    xmaxs=[]
    ymaxs=[]
    _paths=[]
    # images
    for img_path in tqdm(img_paths):
        base=img_path.split(".")[0]
        # text path
        _iden=os.path.basename(img_path).split(".")[0]
        text_path=os.path.join(_dir,coords,f"{_iden}.txt")
        with open(text_path,"r") as tf:
            lines=tf.readlines()
        for line in lines:
            parts=line.split()
            if len(parts)>4:
                line_num=parts[0].replace("\ufeff","")
                word_num=parts[1]
                label=parts[2]
                data=parts[3]
                x,y,w,h=[int(i) for i in parts[-1].split(",")]
                liness.append(line_num)
                words.append(word_num)
                chars.append(label)
                xmins.append(x)
                ymins.append(y)
                xmaxs.append(x+w)
                ymaxs.append(y+h)
                _paths.append(img_path)
                comps.append(data)
    df=pd.DataFrame({"line":liness,
                     "word":words,
                     "char":chars,
                     "comp":comps,
                     "xmin":xmins,
                     "ymin":ymins,
                     "xmax":xmaxs,
                     "ymax":ymaxs,
                     "image":_paths})
    return df

def check_missing(_dir,coords,fmt):
    '''
        checks for missing data
    '''
    img_paths=[img_path for img_path in glob(os.path.join(_dir,f"*.{fmt}"))]
    txt_paths=[txt_path for txt_path in glob(os.path.join(_dir,coords,"*.txt"))]
    # error check
    for img_path in tqdm(img_paths):
        if "jpg" in img_path:
            _iden=os.path.basename(img_path).split(".")[0]
            txt_path=os.path.join(_dir,coords,f"{_iden}.txt")
            if not os.path.exists(txt_path):
                print(img_path)
                for txt in txt_paths:
                    if _iden in txt:
                        print(txt)
                        niden=os.path.basename(txt).split('.')[0]
                        print(f"RENAME:{_iden} to {niden}")
                        os.rename(os.path.join(_dir,f"{_iden}.{fmt}"),
                                  os.path.join(_dir,f"{niden}.{fmt}"))
                        
                        

def main(args):
    readme_txt_path=args.readme_txt_path
    save_path=args.save_path

    main_path=create_dir(save_path,"bs")
    save_path=create_dir(main_path,"images")

    base_path=os.path.dirname(readme_txt_path)
    LOG_INFO(base_path)
    assert len(os.listdir(base_path))==5,"WRONG PATH FOR README.txt"

    os.listdir(base_path)
    dfs=[]
    # ## 1.Camera
    _dir=os.path.join(base_path,'1. Camera','1. Essay')
    coords='Character Coordinates_a'
    fmt="jpg"
    check_missing(_dir,coords,fmt)
    dfs.append(extract_info(_dir,coords,fmt))
    # ## 2. Scan
    _dir=os.path.join(base_path,'2. Scan','1. Essay')
    coords='Character Coordinates_a'
    fmt="tif"
    check_missing(_dir,coords,fmt)
    dfs.append(extract_info(_dir,coords,fmt))
    # # 3. Conjunct
    _dir=os.path.join(base_path,'3. Conjunct')
    coords='Character Coordinates'
    fmt="tif"
    check_missing(_dir,coords,fmt)
    dfs.append(extract_info(_dir,coords,fmt))
    df=pd.concat(dfs,ignore_index=True)
    



    filepath=[]
    words=[]
    source=[]
    iden=0
    for img_path in tqdm(df.image.unique()):
        idf=df.loc[df.image==img_path]
        #-------------
        # image
        #-------------
        img=cv2.imread(img_path)
        # charmap
        cimg=removeShadow(img)
        cimg=cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cimg,(5,5),0)
        _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        for line in idf.line.unique():
            linedf=idf.loc[idf.line==line]
            for word in linedf.word.unique():
                wdf=linedf.loc[linedf.word==word]
                # word
                xmin=int(min(wdf.xmin.tolist()))
                xmax=int(max(wdf.xmax.tolist()))

                ymin=int(min(wdf.ymin.tolist()))
                ymax=int(max(wdf.ymax.tolist()))

                data=img[ymin:ymax,xmin:xmax]
                data=stripPads(data,255)
                
                fname=f"{iden}.png"
                cv2.imwrite(os.path.join(save_path,fname),data)
                filepath.append(os.path.join(save_path,fname))
                words.append("".join(wdf.comp.tolist()))
                source.append(img_path.replace(base_path,""))
                iden+=1
        

    data=pd.DataFrame({"filepath":filepath,"word":words,"source":source})
    data.dropna(inplace=True)
    data.to_csv(os.path.join(main_path,"data.csv"),index=False)

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("Boise State Dataset Creating Script")
    parser.add_argument("readme_txt_path", help="Path of the readme_text file present within the dataset that holds folders such as camera,conjuct etc")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    args = parser.parse_args()
    main(args)


