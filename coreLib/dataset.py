# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import os
import pandas as pd 
from ast import literal_eval
from glob import glob
from tqdm import tqdm
from .utils import LOG_INFO
tqdm.pandas()
#--------------------
# class info
#--------------------
class bangla:
    vowels                 =   ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']
    consonants             =   ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 
                                'চ', 'ছ','জ', 'ঝ', 'ঞ', 
                                'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 
                                'ত', 'থ', 'দ', 'ধ', 'ন', 
                                'প', 'ফ', 'ব', 'ভ', 'ম', 
                                'য', 'র', 'ল', 'শ', 'ষ', 
                                'স', 'হ','ড়', 'ঢ়', 'য়']
    modifiers              =   ['ঁ', 'ং', 'ঃ','ৎ']
    # diacritics
    vowel_diacritics       =   ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
    consonant_diacritics   =   ['ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']
    # special charecters
    nukta                  =   '়'
    hosonto                =   '্'
    special_charecters     =   [ nukta, hosonto,'\u200d']
    
    mods=['ঁ', 'ং', 'ঃ']

    top_exts               =   ['ই', 'ঈ', 'উ', 'ঊ', 'ঐ','ঔ','ট', 'ঠ',' ি', 'ী', 'ৈ', 'ৌ','ঁ','র্']

    bot_exts               =  ['ু', 'ূ', 'ৃ',]
    valid                  =  vowels+consonants+vowel_diacritics+consonant_diacritics+special_charecters
        
class DataSet(object):
    def __init__(self,data_dir):
        '''
            data_dir : the location of the data folder
        '''
        self.data_dir       =   data_dir
            
            
        class graphemes:
            dir   =   os.path.join(data_dir,"bangla","graphemes")
            csv   =   os.path.join(data_dir,"bangla","graphemes.csv")

        self.all_fonts       =   [fpath for fpath in glob(os.path.join(data_dir,"bangla","fonts","*.ttf"))]

        # assign
        self.graphemes          = graphemes
        # error check
        self.__checkExistance()        
        # get df
        self.graphemes.df    =self.__getDataFrame(self.graphemes.csv)
        
        # graphemes
        self.known_graphemes=sorted(list(self.graphemes.df.label.unique()))
        # data validity
        self.__checkDataValidity(self.graphemes,"bangla.graphemes")
        

    def __checkExistance(self):
        '''
            check for paths and make sure the data is there 
        '''
        assert os.path.exists(self.graphemes.dir),"Bangla graphemes dir not found"
        assert os.path.exists(self.graphemes.csv),"Bangla graphemes csv not found" 
        LOG_INFO("All paths found",mcolor="green")
    

    def __getDataFrame(self,csv,is_dict=False):
        '''
            creates the dataframe from a given csv file
            args:
                csv       =   csv file path
                is_dict   =   if the csv is a dictionary
        '''
        try:
            df=pd.read_csv(csv)
            if is_dict:
                assert "word" in df.columns,f"word column not found:{csv}"
                assert "graphemes" in df.columns,f"graphemes column not found:{csv}"
                df.graphemes=df.graphemes.progress_apply(lambda x: literal_eval(x))
                df= df.sample(frac=1)
            else:    
                assert "filename" in df.columns,f"filename column not found:{csv}"
                assert "label" in df.columns,f"label column not found:{csv}"
                df.label=df.label.progress_apply(lambda x: str(x))
            return df
        except Exception as e:
            LOG_INFO(f"Error in processing:{csv}",mcolor="yellow")
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
