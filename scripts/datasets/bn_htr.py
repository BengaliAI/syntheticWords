# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import sys
sys.path.append('../')
import argparse
import os 
import cv2
import pandas as pd 
from glob import glob
from tqdm.auto import tqdm
from coreLib.utils import *
tqdm.pandas()

def get_labels(data_path):
    '''
        extract labels and sources
    '''
    dfs=[]
    for i in tqdm(range(1,151)):
        if i==60:# Exception
            xlsx=os.path.join(data_path,f"{i}",f"{i}.xl.xlsx")
        else:
            xlsx=os.path.join(data_path,f"{i}",f"{i}.xlsx")
        df=pd.read_excel(xlsx)
        if len(df.columns)==0:
            df=pd.read_excel(xlsx,sheet_name='Sheet1')
        
        if "Id" in df.columns:
            filename=df["Id"].tolist()
        else:
            filename=df["ID"].tolist()

        if "Word" in df.columns:
            labels=df["Word"].tolist()
        else:
            labels=df["word"].tolist()

        df=pd.DataFrame({"source":filename,"word":labels})
        dfs.append(df)
    # valid idens
    df=pd.concat(dfs,ignore_index=True)
    idens=df["source"].tolist()
    valid=[]
    for i in tqdm(range(1,151)):
        folder=os.path.join(data_path,f"{i}","Words")
        img_paths=[img_path for img_path in glob(os.path.join(folder,"*/*.*"))]
        for src in img_paths:
            base=os.path.basename(src).split(".")[0]
            if base in idens:
                valid.append(src)

    return df,valid

def main(args):
    data_path=args.data_path
    save_path=args.save_path
    main_path=create_dir(save_path,"bh")
    save_path=create_dir(main_path,"images")
    
    assert len(os.listdir(data_path))==150,"WORNG data_path for folders"

    iden=0
    filepath=[]
    word=[]
    source=[]
    df,valid=get_labels(data_path)

    for img_path in tqdm(valid):
        img=cv2.imread(img_path)
        
        # base
        base=os.path.basename(img_path).split(".")[0]
        idf=df.loc[df["source"]==base]
        _word=idf.word.tolist()[0]
        _source=idf["source"].tolist()[0].split("_")[0]
        fname=f"{iden}.png"
        
        cv2.imwrite(os.path.join(save_path,fname),img)
        filepath.append(os.path.join(save_path,fname))
        word.append(_word)
        source.append(_source)
        iden+=1
        
    data  =   pd.DataFrame({"filepath":filepath,"word":word,"source":source})
    data.dropna(inplace=True)
    data.to_csv(os.path.join(main_path,"data.csv"),index=False)


#-----------------------------------------------------------------------------------

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("BN-HTRd Dataset Creating Script")
    parser.add_argument("data_path", help="Path of the data folder that contains 1 to 150 named folders")
    parser.add_argument("save_path", help="Path of the directory to save the dataset")
    args = parser.parse_args()
    main(args)

