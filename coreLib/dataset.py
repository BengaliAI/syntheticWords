# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd 
from glob import glob
from tqdm import tqdm
from .utils import *
tqdm.pandas()
#--------------------
# class info
#--------------------

        
class DataSet(object):
    def __init__(self,data_dir,language,use_printed_only=False):
        '''
            data_dir : the location of the data folder
        '''
        self.data_dir       =   data_dir
        self.language       =   language    
        if not use_printed_only:
            # resources
            class graphemes:
                dir   =   os.path.join(data_dir,language,"graphemes")
                csv   =   os.path.join(data_dir,language,"graphemes.csv")

            class numbers:
                dir   =   os.path.join(data_dir,language,"numbers")
                csv   =   os.path.join(data_dir,language,"numbers.csv")

        
        self.fonts       =   [fpath for fpath in glob(os.path.join(data_dir,language,"fonts","*.ttf"))]
        if not use_printed_only:
            # assign
            self.graphemes          = graphemes
            self.numbers            = numbers
            # error check
            self.__checkExistance()        
            # get df
            self.graphemes.df    =  self.__getDataFrame(self.graphemes)
            self.numbers.df      =  self.__getDataFrame(self.numbers)
            # data validity
            self.__checkDataValidity(self.graphemes,f"{self.language}.graphemes")
            self.__checkDataValidity(self.numbers,f"{self.language}.numbers")
            # lists
            self.graphemes_list= sorted(list(self.graphemes.df.label.unique()))
            self.numbers_list  = sorted(list(self.numbers.df.label.unique()))
            # combined data
            self.valid_graphemes=sorted(self.graphemes_list+self.numbers_list)
            self.df             =pd.concat([self.graphemes.df,self.numbers.df],ignore_index=False)

    def __checkExistance(self):
        '''
            check for paths and make sure the data is there 
        '''
        assert os.path.exists(self.graphemes.dir),f"{self.language} graphemes dir not found"
        assert os.path.exists(self.graphemes.csv),f"{self.language} graphemes csv not found" 
        assert os.path.exists(self.numbers.dir),f"{self.language} numbers dir not found"
        assert os.path.exists(self.numbers.csv),f"{self.language} numbers csv not found" 
        LOG_INFO("All paths found",mcolor="green")
    

    def __getDataFrame(self,obj):
        '''
            creates the dataframe from a given csv file
            args:
                obj       =   the obj that has csv and dir
                
        '''
        try:
            df=pd.read_csv(obj.csv)
            assert "filename" in df.columns,f"filename column not found:{obj.csv}"
            assert "label" in df.columns,f"label column not found:{obj.csv}"
            df.label=df.label.progress_apply(lambda x: str(x))
            df["img_path"]=df["filename"].progress_apply(lambda x:os.path.join(obj.dir,f"{x}.bmp"))
            return df
        except Exception as e:
            LOG_INFO(f"Error in processing:{obj.csv}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red") 
                

    def __checkDataValidity(self,obj,iden):
        '''
            checks that a folder does contain proper images
        '''
        try:
            LOG_INFO(iden)
            imgs=[img_path for img_path in tqdm(glob(os.path.join(obj.dir,"*.*")))]
            assert len(imgs)>0, f"No data paths found({iden})"
            assert len(imgs)==len(obj.df), f"Image paths doesnot match label data({iden}:{len(imgs)}!={len(obj.df)})"
            
        except Exception as e:
            LOG_INFO(f"Error in Validity Check:{iden}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red")                
