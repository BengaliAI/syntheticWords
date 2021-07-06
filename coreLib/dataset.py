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
from .utils import create_dir,LOG_INFO
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
        
        #-------------------
        # save_paths
        #------------------        
        self.save_path                  =   create_dir(save_path,"data")
        self.images_path                =   create_dir(self.save_path,"images")
        self.bangla_writing_path        =   create_dir(self.images_path,"bangla_writing")       # bangla_writing data
        self.boise_state_path           =   create_dir(self.images_path,"boise_state")          # boise state data
        self.bn_htr_path                =   create_dir(self.images_path,"bn_htr")               # bn htr data
        self.synthetic_path             =   create_dir(self.images_path,"synthetic")            # synthetic data 

        # csv-s
        self.bangla_writing_csv         =   os.path.join(self.save_path,"bangla_writing.csv")   # intermediate saving
        self.bangla_writing_train_csv           =   os.path.join(self.save_path,"bangla_writing_train.csv")
        self.bangla_writing_eval_csv            =   os.path.join(self.save_path,"bangla_writing_eval.csv")
        
        self.boise_state_csv            =   os.path.join(self.save_path,"boise_state.csv")
        self.bn_htr_csv                 =   os.path.join(self.save_path,"bn_htr.csv")
        self.synth_csv                  =   os.path.join(self.save_path,"synthetic.csv")

        self.config_json    =   os.path.join(self.save_path,"config.json")
        
        
        class tfrecords:
            dir                 =   create_dir(self.save_path,"tfrecords")
            synthetic           =   create_dir(dir,"synthetic")
            bn_htr              =   create_dir(dir,"bn_htr")
            boise_state         =   create_dir(dir,"boise_state")  
            # split case      
            bangla_writing      =   create_dir(dir,"bangla_writing")
            train_bw            =   create_dir(bangla_writing,"train")
            eval_bw             =   create_dir(bangla_writing,"eval")
            

        #-------------------
        # resource
        #------------------
        class graphemes:
            dir   =   os.path.join(data_dir,"graphemes")
            csv   =   os.path.join(data_dir,"graphemes.csv")
        
        self.pages       =   os.path.join(data_dir,"bangla_writing")
        
        class boise_state:
            dir =   os.path.join(data_dir,"boise_state","words")
            csv =   os.path.join(data_dir,"boise_state","labels.csv")
        
        class bn_htr:
            dir =   os.path.join(data_dir,"bn_htr","words")
            csv =   os.path.join(data_dir,"bn_htr","labels.csv")
        
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

            # all valid unicode charecters
            valid_unicodes         =    vowels+ consonants+ modifiers+ vowel_diacritics+ special_charecters





        # assign
        self.graphemes  = graphemes
        self.tfrecords  = tfrecords
        self.boise_state= boise_state
        self.bangla     = bangla
        self.bn_htr     = bn_htr  

        # error check
        self.__checkExistance()

        # get df
        self.graphemes.df   =self.__getDataFrame(self.graphemes.csv)
        self.boise_state.df =self.__getDataFrame(self.boise_state.csv,label_type="list")
        self.bn_htr.df      =self.__getDataFrame(self.bn_htr.csv,label_type="list")
        
        # data validity
        self.__checkDataValidity(self.graphemes,"graphemes")
        self.__checkDataValidity(self.pages,"pages",check_pages=True)
        self.__checkDataValidity(self.boise_state,"graphemes")
        self.__checkDataValidity(self.bn_htr,"graphemes",extra=True)
        
        # graphemes
        self.known_graphemes=list(self.graphemes.df.label.unique())

    def __getDataFrame(self,csv,label_type="single"):
        '''
            creates the dataframe from a given csv file
            args:
                csv       =   csv file path
                label_type=   either single label or list of label is given. For non-single anything goes
        '''
        try:
            df=pd.read_csv(csv)
            assert "filename" in df.columns,f"filename column not found:{csv}"
            if label_type=="single":
                assert "label" in df.columns,f"label column not found:{csv}"
            else:
                assert "labels" in df.columns,f"label column not found:{csv}"
                df.labels=df.labels.progress_apply(lambda x: literal_eval(x))
            return df
        except Exception as e:
            LOG_INFO(f"Error in processing:{csv}",mcolor="yellow")
            LOG_INFO(f"{e}",mcolor="red") 
                

    def __checkDataValidity(self,obj,iden,check_pages=False,extra=False):
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
                if not extra:
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
        assert os.path.exists(self.boise_state.dir),"Boise State Image Dir not found"
        assert os.path.exists(self.boise_state.csv),"Boise State csv not found"
        assert os.path.exists(self.bn_htr.dir),"BN HTR State Image Dir not found"
        assert os.path.exists(self.bn_htr.csv),"BN HTR csv not found"
        
        LOG_INFO("All paths found",mcolor="green")
        
        
        
        
        


    
        