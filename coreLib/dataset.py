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
from tqdm.auto import tqdm
from .utils import *
tqdm.pandas()
#--------------------
# class info
#--------------------
class DataSet(object):
    def __init__(self,data_dir,save_path):
        '''
            data_dir : the location of the data folder
            save_path: the path to save all outputs
        '''
        self.data_dir=data_dir
        # save paths
        self.save_path      =   save_path
        self.word_path      =   create_dir(self.save_path,"words")
        self.synthetic_path =   create_dir(self.save_path,"synthetic")
        self.config_json    =   os.path.join(self.save_path,"config.json")
        class tfrecords:
            dir         =   create_dir(self.save_path,"tfrecords")
            synthetic   =   create_dir(dir,"synthetic")
            written     =   create_dir(dir,"written")
            train       =   create_dir(written,"train")
            test        =   create_dir(written,"test")        


        class graphemes:
            dir   =   os.path.join(data_dir,"graphemes")
            csv   =   os.path.join(data_dir,"graphemes.csv")
        
        self.pages     =   os.path.join(data_dir,"pages")
    
        

        # assign
        self.graphemes  = graphemes
        self.tfrecords  = tfrecords
        # error check
        self.__checkExistance()

        # get df
        self.graphemes.df=self.__getDataFrame(self.graphemes.csv)
        
        # data validity
        self.__checkDataValidity(self.graphemes,"graphemes")
        self.__checkDataValidity(self.pages,"pages",check_pages=True)
        # graphemes
        self.known_graphemes=list(self.graphemes.df.label.unique())

    def __getDataFrame(self,csv):
        '''
            creates the dataframe from a given csv file
            args:
                csv       =   csv file path
        '''
        try:
            df=pd.read_csv(csv)
            assert "filename" in df.columns,f"filename column not found:{csv}"
            assert "label" in df.columns,f"label column not found:{csv}"
            return df
        except Exception as e:
            LOG_INFO(f"Error in processing:{csv}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red") 
                

    def __checkDataValidity(self,obj,iden,check_pages=False):
        '''
            checks that a folder does contain proper images
        '''
        try:
            LOG_INFO(iden)
            if check_pages:
                imgs =[data_path for data_path in tqdm(glob(os.path.join(obj,"*.jpg*")))]
                jsons=[data_path for data_path in tqdm(glob(os.path.join(obj,"*.json*")))]
                assert len(imgs)==len(jsons), "Image and Annotation Mismatch For pages"
            else:
                imgs=[img_path for img_path in tqdm(glob(os.path.join(obj.dir,"*.*")))]
                assert len(imgs)>0, f"No data paths found({iden})"
                assert len(imgs)==len(obj.df), f"Image paths doesnot match labels data({iden}:{len(imgs)}!={len(obj.df)})"
                
        except Exception as e:
            LOG_INFO(f"Error in Validity Check:{iden}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red")        


    def __checkExistance(self):
        '''
            check for paths and make sure the data is there 
        '''
        assert os.path.exists(self.graphemes.dir),"Bangla graphemes dir not found"
        assert os.path.exists(self.graphemes.csv),"Bangla graphemes csv not found"
        assert os.path.exists(self.pages),"Pages dir not found"
        
        LOG_INFO("All paths found",mcolor="green")
        
        
        
        
        


    
        