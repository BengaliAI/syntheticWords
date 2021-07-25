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
class DataSet(object):
    def __init__(self,data_dir):
        '''
            data_dir : the location of the data folder
        '''
        self.data_dir=data_dir
        
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
            punctuations           =   ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
                                        ',', '-', '.', '/', ':', ':-', ';', '<', '=', '>', '?', 
                                        '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '।', '—', '’', '√']

            number_values          =    ['০','১','২','৩','৪','৫','৬','৭','৮','৯']

            mods=['ঁ', 'ং', 'ঃ']

            top_exts               =   ['ই', 'ঈ', 'উ', 'ঊ', 'ঐ','ঔ','ট', 'ঠ',' ি', 'ী', 'ৈ', 'ৌ','ঁ','র্']

            bot_exts               =  ['ু', 'ূ', 'ৃ',]

            class graphemes:
                dir   =   os.path.join(data_dir,"bangla","graphemes")
                csv   =   os.path.join(data_dir,"bangla","graphemes.csv")

            class numbers:
                dir   =   os.path.join(data_dir,"bangla","numbers")
                csv   =   os.path.join(data_dir,"bangla","numbers.csv")

            dictionary_csv  =   os.path.join(data_dir,"bangla","dictionary.csv")    
            font            =   os.path.join(data_dir,"bangla","fonts","Bangla.ttf")


        # assign
        self.bangla     = bangla
        # error check
        self.__checkExistance()
        # get df
        self.bangla.graphemes.df    =self.__getDataFrame(self.bangla.graphemes.csv)
        self.bangla.numbers.df      =self.__getDataFrame(self.bangla.numbers.csv)
        self.bangla.dictionary      =self.__getDataFrame(self.bangla.dictionary_csv,is_dict=True)
        

        # graphemes
        self.known_graphemes=sorted(list(self.bangla.graphemes.df.label.unique()))
        # cleanup
        self.bangla.dictionary.graphemes    =   self.bangla.dictionary.graphemes.progress_apply(lambda x: x if set(x)<=set(self.known_graphemes) else None)
        self.bangla.dictionary.dropna(inplace=True)
        
        
        
        # data validity
        self.__checkDataValidity(self.bangla.graphemes,"bangla.graphemes")
        self.__checkDataValidity(self.bangla.numbers,"bangla.numbers")
        
        # vocab graphemes
        self.bangla.gvocab=[""]+self.bangla.punctuations+self.bangla.number_values+self.known_graphemes

    def __checkExistance(self):
        '''
            check for paths and make sure the data is there 
        '''
        assert os.path.exists(self.bangla.graphemes.dir),"Bangla graphemes dir not found"
        assert os.path.exists(self.bangla.graphemes.csv),"Bangla graphemes csv not found"
        assert os.path.exists(self.bangla.numbers.dir),"Bangla numbers dir not found"
        assert os.path.exists(self.bangla.numbers.csv),"Bangla numbers csv not found"
        assert os.path.exists(self.bangla.dictionary_csv),"Bangla dictionary csv not found"
        assert os.path.exists(self.bangla.font),"Bangla.ttf font not found"
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