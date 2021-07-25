#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from numpy.lib.financial import pv
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
from tqdm import tqdm
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path

#---------------------------------------------------------------
# image utils
#---------------------------------------------------------------
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


def padImage(img,pad_loc,pad_dim,pad_type,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_type:       central or left aligned pad
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w=img.shape
        if pad_type=="central":
            # pad widths
            left_pad_width =(pad_dim-w)//2
            # print(left_pad_width)
            right_pad_width=pad_dim-w-left_pad_width
            # pads
            left_pad =np.ones((h,left_pad_width))*pad_val
            right_pad=np.ones((h,right_pad_width))*pad_val
            # pad
            img =np.concatenate([left_pad,img,right_pad],axis=1)
        else:
            # pad widths
            pad_width =pad_dim-w
            # pads
            pad =np.ones((h,pad_width))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8")    

def correctPadding(img,dim,ptype="central",pvalue=255):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            ptype   :       type of padding (central,left)
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w=img.shape
    
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padImage(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=0

    elif w < img_width:
        # pad
        img=padImage(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        mask=w
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 

#----------------------------------------------------------------
# Dataset utils
#----------------------------------------------------------------

def extend_vocab(symbol_lists,curr_vocab):
    '''
        creates a sorted vocabulary list 
        args:
            symbol_lists    :   list of list of symbols
            curr_vocab      :   current vocab
    '''
    vocab=[]
    for symbol_list in tqdm(symbol_lists):
            vocab+=symbol_list
    vocab=sorted(list(set(vocab)))
    
    new_vocab=[]
    # ACCOUNT FOR BLANK INDEX
    for v in vocab:
        if v not in curr_vocab:
            new_vocab.append(v)
    
    return new_vocab
#---------------------------------------------------------------
def get_encoded_label(symbol_list,vocab):
    '''
        creates encoded label for images (multihot encoding)
        args:
            symbol_list   :   the list of symbols
            vocab         :   the total list of vocabulary 
    '''
    encoded=[]
    for symbol in symbol_list:
        encoded.append(vocab.index(symbol))
    return encoded
#---------------------------------------------------------------
def pad_encoded_label(label,max_len,pad_value):
    '''
        pad an encoded label
        args:
            label       :   the encoded label to pad
            max_len     :   the maximum lenth to pad
            pad_value   :   the value to pad 
    '''
    for _ in range(max_len-len(label)):
        label.append(pad_value)
    return label
    
#----------------------------------------------------------------
# Parser util
#----------------------------------------------------------------
class GraphemeParser(object):
    '''
    @author: Tahsin Reasat
    Adoptation:MD. Nazmuddoha Ansary
    '''
    def __init__(self):
        self.vds    =['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
        self.cds    =['ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']
        self.roots  =['ং','ঃ','অ','আ','ই','ঈ','উ','ঊ','ঋ','এ','ঐ','ও','ঔ','ক','ক্ক','ক্ট','ক্ত','ক্ল','ক্ষ','ক্ষ্ণ',
                    'ক্ষ্ম','ক্স','খ','গ','গ্ধ','গ্ন','গ্ব','গ্ম','গ্ল','ঘ','ঘ্ন','ঙ','ঙ্ক','ঙ্ক্ত','ঙ্ক্ষ','ঙ্খ','ঙ্গ','ঙ্ঘ','চ','চ্চ',
                    'চ্ছ','চ্ছ্ব','ছ','জ','জ্জ','জ্জ্ব','জ্ঞ','জ্ব','ঝ','ঞ','ঞ্চ','ঞ্ছ','ঞ্জ','ট','ট্ট','ঠ','ড','ড্ড','ঢ','ণ',
                    'ণ্ট','ণ্ঠ','ণ্ড','ণ্ণ','ত','ত্ত','ত্ত্ব','ত্থ','ত্ন','ত্ব','ত্ম','থ','দ','দ্ঘ','দ্দ','দ্ধ','দ্ব','দ্ভ','দ্ম','ধ',
                    'ধ্ব','ন','ন্জ','ন্ট','ন্ঠ','ন্ড','ন্ত','ন্ত্ব','ন্থ','ন্দ','ন্দ্ব','ন্ধ','ন্ন','ন্ব','ন্ম','ন্স','প','প্ট','প্ত','প্ন',
                    'প্প','প্ল','প্স','ফ','ফ্ট','ফ্ফ','ফ্ল','ব','ব্জ','ব্দ','ব্ধ','ব্ব','ব্ল','ভ','ভ্ল','ম','ম্ন','ম্প','ম্ব','ম্ভ',
                    'ম্ম','ম্ল','য','র','ল','ল্ক','ল্গ','ল্ট','ল্ড','ল্প','ল্ব','ল্ম','ল্ল','শ','শ্চ','শ্ন','শ্ব','শ্ম','শ্ল','ষ',
                    'ষ্ক','ষ্ট','ষ্ঠ','ষ্ণ','ষ্প','ষ্ফ','ষ্ম','স','স্ক','স্ট','স্ত','স্থ','স্ন','স্প','স্ফ','স্ব','স্ম','স্ল','স্স','হ',
                    'হ্ন','হ্ব','হ্ম','হ্ল','ৎ','ড়','ঢ়','য়']

        self.punctuations           =   ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
                                        ',', '-', '.', '/', ':', ':-', ';', '<', '=', '>', '?', 
                                        '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '।', '—', '’', '√']

        self.numbers                =    ['০','১','২','৩','৪','৫','৬','৭','৮','৯']
        self.ignore                 =   self.punctuations+self.numbers


    def word2grapheme(self,word):
        graphemes = []
        grapheme = ''
        i = 0
        while i < len(word):
            if word[i] in self.ignore:
                graphemes.append(word[i])
            else:
                grapheme += (word[i])
                # print(word[i], grapheme, graphemes)
                # deciding if the grapheme has ended
                if word[i] in ['\u200d', '্']:
                    # these denote the grapheme is contnuing
                    pass
                elif word[i] == 'ঁ':  
                    # 'ঁ' always stays at the end
                    graphemes.append(grapheme)
                    grapheme = ''
                elif word[i] in list(self.roots) + ['়']:
                    # root is generally followed by the diacritics
                    # if there are trailing diacritics, don't end it
                    if i + 1 == len(word):
                        graphemes.append(grapheme)
                    elif word[i + 1] not in ['্', '\u200d', 'ঁ', '়'] + list(self.vds):
                        # if there are no trailing diacritics end it
                        graphemes.append(grapheme)
                        grapheme = ''

                elif word[i] in self.vds:
                    # if the current character is a vowel diacritic
                    # end it if there's no trailing 'ঁ' + diacritics
                    # Note: vowel diacritics are always placed after consonants
                    if i + 1 == len(word):
                        graphemes.append(grapheme)
                    elif word[i + 1] not in ['ঁ'] + list(self.vds):
                        graphemes.append(grapheme)
                        grapheme = ''

            i = i + 1
            # Note: df_cd's are constructed by df_root + '্'
            # so, df_cd is not used in the code

        return graphemes

    

#-------------------------------------------
# cleaner class
#-------------------------------------------
class WordCleaner(object):
    def __init__(self):
        # components    
        '''
            this division of vowel, consonant and modifier is done according to :https://bn.wikipedia.org/wiki/%E0%A7%8E 
        '''
        self.vowels                 =   ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']
        self.consonants             =   ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 
                                         'চ', 'ছ','জ', 'ঝ', 'ঞ', 
                                         'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 
                                         'ত', 'থ', 'দ', 'ধ', 'ন', 
                                         'প', 'ফ', 'ব', 'ভ', 'ম', 
                                         'য', 'র', 'ল', 'শ', 'ষ', 
                                         'স', 'হ','ড়', 'ঢ়', 'য়']
        self.modifiers              =   ['ঁ', 'ং', 'ঃ','ৎ']
        # diacritics
        self.vowel_diacritics       =   ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
        self.consonant_diacritics   =   ['ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']
        # special charecters
        self.nukta                  =   '়'
        self.hosonto                =   '্'
        self.special_charecters     =   [self.nukta,self.hosonto,'\u200d']
        
        
        self.punctuations           =   ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
                                        ',', '-', '.', '/', ':', ':-', ';', '<', '=', '>', '?', 
                                        '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '।', '—', '’', '√']

        self.numbers                =    ['০','১','২','৩','৪','৫','৬','৭','৮','৯']
        
        # all valid unicode charecters
        self.valid_unicodes         =   self.vowels+self.consonants+self.modifiers+self.vowel_diacritics+self.special_charecters+self.numbers+self.punctuations
        
        '''
            some cases to handle
        '''
        
        # invalid unicodes for starting
        '''
            no vowel diacritic, consonant diacritic , special charecter or modifier can start a word
        '''   
        self.invalid_unicodes_for_starting_a_word=self.modifiers+self.vowel_diacritics+self.special_charecters+self.consonant_diacritics
        
        
        
        # invalid hosonto cases
        '''
            a hosonto can not come before:
                * the vowels
                * another hosonto [double consecutive hosonto]
            a hosonto can not come after:
                * the vowels
                * the modifiers
                * another hosonto [double consecutive hosonto] 
        '''
        self.invalid_unicodes_after_hosonto     =       self.vowels+[self.hosonto]
        self.invalid_unicodes_before_hosonto    =       self.vowels+self.modifiers+[self.hosonto]
        
        
        
        # to+hosonto case
        '''
            case-1:     if 'ত'+hosonto is followed by anything other than a consonant the word is an invalid word
            case-2:     The ত্‍ symbol which should be replaced by a 'ৎ' occurs for all consonants except:ত,থ,ন,ব,ম,য,র
                        # code to verify this manually 
                        for c in self.consonants:
                            print('ত'+self.hosonto+c)
 
        '''
        self.valid_consonants_after_to_and_hosonto      =       ['ত','থ','ন','ব','ম','য','র'] 
       

    def __replaceDiacritics(self):
        '''
            case: replace  diacritic 
                # Example-1: 
                (a)'আরো'==(b)'আরো' ->  False 
                    (a) breaks as:['আ', 'র', 'ে', 'া']
                    (b) breaks as:['আ', 'র', 'ো']
                # Example-2:
                (a)পৌঁছে==(b)পৌঁছে ->  False
                    (a) breaks as:['প', 'ে', 'ৗ', 'ঁ', 'ছ', 'ে']
                    (b) breaks as:['প', 'ৌ', 'ঁ', 'ছ', 'ে']
                # Example-3:
                (a)সংস্কৄতি==(b)সংস্কৃতি ->  False
                    (a) breaks as:['স', 'ং', 'স', '্', 'ক', 'ৄ', 'ত', 'ি']
                    (b) breaks as:['স', 'ং', 'স', '্', 'ক', 'ৃ', 'ত', 'ি']
                
                            
        '''
        # broken vowel diacritic
        # e-kar+a-kar = o-kar
        self.word = self.word.replace('ে'+'া', 'ো')
        # e-kar+e-kar = ou-kar
        self.word = self.word.replace('ে'+'ৗ', 'ৌ')
        # 'অ'+ 'া'-->'আ'
        self.word = self.word.replace('অ'+ 'া','আ')
        # unicode normalization of 'ৄ'-> 'ৃ'
        self.word = self.word.replace('ৄ','ৃ')
        
    def __createDecomp(self):
        '''
            create list of valid unicodes
        '''
        self.decomp=[ch for ch in self.word if ch in self.valid_unicodes]
        if not self.__checkDecomp():
            self.return_none=True

    def __checkDecomp(self):
        '''
            checks if the decomp has a valid length
        '''
        if len(self.decomp)>0:
            return True
        else:
            return False

            

    def __cleanInvalidEnds(self):
        '''
            cleans a word that has invalid ending i.e ends with '্' that does not make any sense
        '''
        while self.decomp[-1] == self.hosonto:
            self.decomp=self.decomp[:-1]
            if not self.__checkDecomp():
                self.return_none=True
                break 


    def __cleanInvalidStarts(self):
        '''
            cleans a word that has invalid starting
        '''
        while self.decomp[0] in self.invalid_unicodes_for_starting_a_word:
            self.decomp=self.decomp[1:]
            if not self.__checkDecomp():
                self.return_none=True
                break 

            

    def __cleanNuktaUnicode(self):
        '''
            handles nukta unicode as follows:
                * If the connecting char is with in the valid list ['য','ব','ড','ঢ'] then replace with ['য়','র','ড়', 'ঢ়']
                * Otherwise remove the nukta char completely
            **the connecting char**: is defined as the previous non-vowle-diacritic char 
            Example-1:If case-1
            (a)কেন্দ্রীয়==(b)কেন্দ্রীয় ->  False
                (a) breaks as:['ক', 'ে', 'ন', '্', 'দ', '্', 'র', 'ী', 'য', '়']
                (b) breaks as:['ক', 'ে', 'ন', '্', 'দ', '্', 'র', 'ী', 'য়']
            Example-2:Elif case-2
            (a)রযে়ছে==(b)রয়েছে ->  False
                (a) breaks as:['র', 'য', 'ে', '়', 'ছ', 'ে']
                (b) breaks as:['র', 'য়', 'ে', 'ছ', 'ে']
            Example-3:Otherwise 
            (a)জ়ন্য==(b)জন্য ->  False
                (a) breaks as:['জ', '়', 'ন', '্', 'য']
                (b) breaks as:['জ', 'ন', '্', 'য']
        '''            
        __valid_charecters_without_nukta    =   ['য','ব','ড','ঢ']
        __replacements                      =   ['য়','র','ড়','ঢ়']
        try:
            for idx,d in enumerate(self.decomp):
                if d==self.nukta:
                    check=False
                    # check the previous charecter is a valid charecter where the nukta can be added
                    if self.decomp[idx-1] in __valid_charecters_without_nukta:
                        cid=idx-1
                        check=True
                    # check the previous char before vowel diacritic
                    elif self.decomp[idx-2] in __valid_charecters_without_nukta and self.decomp[idx-1] in self.vowel_diacritics:
                        cid=idx-2
                        check=True
                    # remove unwanted extra nukta 
                    else:
                        self.decomp[idx]=None
                    if check:
                        rep_char_idx=__valid_charecters_without_nukta.index(self.decomp[cid])
                        # replace
                        self.decomp[cid]=__replacements[rep_char_idx]
                        # delete nukta
                        self.decomp[idx]=None
                              
        except Exception as e:
            pass

    def __cleanInvalidHosonto(self):
        '''
            case:take care of the in valid hosontos that come after / before the vowels and the modifiers
            # Example-1:
            (a)দুই্টি==(b)দুইটি-->False
                (a) breaks as ['দ', 'ু', 'ই', '্', 'ট', 'ি']
                (b) breaks as ['দ', 'ু', 'ই', 'ট', 'ি']
            # Example-2:
            (a)এ্তে==(b)এতে-->False
                (a) breaks as ['এ', '্', 'ত', 'ে']
                (b) breaks as ['এ', 'ত', 'ে']
            # Example-3:
            (a)নেট্ওয়ার্ক==(b)নেটওয়ার্ক-->False
                (a) breaks as ['ন', 'ে', 'ট', '্', 'ও', 'য়', 'া', 'র', '্', 'ক']
                (b) breaks as ['ন', 'ে', 'ট', 'ও', 'য়', 'া', 'র', '্', 'ক']
            # Example-4:
            (a)এস্আই==(b)এসআই-->False
                (a) breaks as ['এ', 'স', '্', 'আ', 'ই']
                (b) breaks as ['এ', 'স', 'আ', 'ই']
            case:if the hosonto is in between two vowel diacritics  
            # Example-1: 
            (a)'চু্ক্তি'==(b)'চুক্তি' ->  False 
                (a) breaks as:['চ', 'ু', '্', 'ক', '্', 'ত', 'ি']
                (b) breaks as:['চ', 'ু','ক', '্', 'ত', 'ি']
            # Example-2:
            (a)'যু্ক্ত'==(b)'যুক্ত' ->   False
                (a) breaks as:['য', 'ু', '্', 'ক', '্', 'ত']
                (b) breaks as:['য', 'ু', 'ক', '্', 'ত']
            # Example-3:
            (a)'কিছু্ই'==(b)'কিছুই' ->   False
                (a) breaks as:['ক', 'ি', 'ছ', 'ু', '্', 'ই']
                (b) breaks as:['ক', 'ি', 'ছ', 'ু','ই']
        '''
        try:
            for idx,d in enumerate(self.decomp):
                if d==self.hosonto:
                    check=False
                    # before case 
                    if self.decomp[idx-1] in self.invalid_unicodes_before_hosonto and self.decomp[idx+1]!='য':
                        check=True    
                    # after case
                    elif self.decomp[idx+1] in self.invalid_unicodes_after_hosonto:
                        check=True
                    # if the hosonto is in between two vowel diacritics
                    elif self.decomp[idx-1] in self.vowel_diacritics or self.decomp[idx+1] in self.vowel_diacritics:
                        check=True
                    # if the hosonto is after modifier
                    elif self.decomp[idx-1] in self.modifiers:
                        check=True
                    
                    if check:
                        self.decomp[idx]=None
        except Exception as e:
            pass                     
    
    def __cleanInvalidToAndHosonto(self):
        '''
            normalizes to+hosonto for ['ত','থ','ন','ব','ম','য','র'] 
            # Example-1:
            (a)বুত্পত্তি==(b)বুৎপত্তি-->False
                (a) breaks as ['ব', 'ু', 'ত', '্', 'প', 'ত', '্', 'ত', 'ি']
                (b) breaks as ['ব', 'ু', 'ৎ', 'প', 'ত', '্', 'ত', 'ি']
            # Example-2:
            (a)উত্স==(b)উৎস-->False
                (a) breaks as ['উ', 'ত', '্', 'স']
                (b) breaks as ['উ', 'ৎ', 'স']
        '''
        try:
            for idx,d in enumerate(self.decomp):
                # to + hosonto
                if d=='ত' and self.decomp[idx+1]==self.hosonto:
                    # for single case
                    if  self.decomp[idx+2] not in self.valid_consonants_after_to_and_hosonto:
                        # replace
                        self.decomp[idx]='ৎ'
                        # delete
                        self.decomp[idx+1]=None
                        
                    else: 
                        # valid replacement for to+hos double case
                        if self.decomp[idx+2]=='ত' and self.decomp[idx+3]==self.hosonto:
                            if self.decomp[idx+4] not in  ['ব','য','র']:
                                # if the next charecter after the double to+hos+to+hos is with in ['ত','থ','ন','ম'] 
                                # replace
                                self.decomp[idx]='ৎ'
                                # delete
                                self.decomp[idx+1]=None
                            if self.decomp[idx+4]=='র':
                                # delete
                                self.decomp[idx+3]=None
                            
        except Exception as e:
            pass
            

    def __cleanDoubleVowelDiacritics(self):
        '''
            removes unwanted doubles(consecutive doubles):
            case:unwanted doubles  
                # Example-1: 
                (a)'যুুদ্ধ'==(b)'যুদ্ধ' ->  False 
                    (a) breaks as:['য', 'ু', 'ু', 'দ', '্', 'ধ']
                    (b) breaks as:['য', 'ু', 'দ', '্', 'ধ']
                # Example-2:
                (a)'দুুই'==(b)'দুই' ->   False
                    (a) breaks as:['দ', 'ু', 'ু', 'ই']
                    (b) breaks as:['দ', 'ু', 'ই']
                # Example-3:
                (a)'প্রকৃৃতির'==(b)'প্রকৃতির' ->   False
                    (a) breaks as:['প', '্', 'র', 'ক', 'ৃ', 'ৃ', 'ত', 'ি', 'র']
                    (b) breaks as:['প', '্', 'র', 'ক', 'ৃ', 'ত', 'ি', 'র']
            case:invalid consecutive vowel diacritics where they are not the same 
            * since there is no way to ensure which one is right it simply returns none
            
        '''
        try:
            for idx,d in enumerate(self.decomp):
                # case of consecutive vowel diacritics
                if d in self.vowel_diacritics and self.decomp[idx+1] in self.vowel_diacritics:
                    # if they are same delete the current one
                    if d==self.decomp[idx+1]:
                        self.decomp[idx]=None
                    # if they are not same --> the word is in valid
                    else:
                        self.return_none=True
                        break
        except Exception as e:
            pass

    
                
                                
    def __cleanVowelDiacriticsComingAfterVowelsAndModifiers(self):
        '''
            takes care of vowels and modifier followed by vowel diacritics
            # Example-1:
            (a)উুলু==(b)উলু-->False
                (a) breaks as ['উ', 'ু', 'ল', 'ু']
                (b) breaks as ['উ', 'ল', 'ু']
            # Example-2:
            (a)আর্কিওোলজি==(b)আর্কিওলজি-->False
                (a) breaks as ['আ', 'র', '্', 'ক', 'ি', 'ও', 'ো', 'ল', 'জ', 'ি']
                (b) breaks as ['আ', 'র', '্', 'ক', 'ি', 'ও', 'ল', 'জ', 'ি']
            
            Also Normalizes 'এ' and 'ত্র'
            # Example-1:
            (a)একএে==(b)একত্রে-->False
                (a) breaks as ['এ', 'ক', 'এ', 'ে']
                (b) breaks as ['এ', 'ক', 'ত', '্', 'র', 'ে']
            # Example-2:
            (a)একএ==(b)একত্র-->False
                (a) breaks as ['এ', 'ক', 'এ']
                (b) breaks as ['এ', 'ক', 'ত', '্', 'র']
                
        '''
        try:
            # THE WIERDEST THING I HAVE SEEN
            for idx,d in enumerate(self.decomp):
                # single case 
                if d=='এ' and idx>0:
                    self.decomp[idx]='ত'+'্'+'র'
            self.decomp=[ch for ch in self.decomp]
            '''
                 self.decomp[idx-1:idx]='ত', '্', 'র'
                 this replacement does not work 
            '''

            for idx,d in enumerate(self.decomp):
                # if the current one is a VD and the previous char is a modifier or vowel
                if  d in self.vowel_diacritics and self.decomp[idx-1] in self.vowels+self.modifiers:
                    # if the vowel is not 'এ'
                    if self.decomp[idx-1] !='এ':
                        # remove diacritic
                         self.decomp[idx]=None
                    # normalization case
                    else:
                        self.decomp[idx]='ত'+'্'+'র'
            self.decomp=[ch for ch in self.decomp]
        except Exception as e:
            pass 

    def __cleanInvalidMultipleConsonantDiacritics(self):
        '''
            cleans repeated folas
        '''
        try:
            for idx,d in enumerate(self.decomp):
                # if the current one is hosonto and the next one is within ['ব','য','র'] 
                if  d==self.hosonto  and self.decomp[idx+1] in ['ব','য','র']:
                    _pair=self.decomp[idx+1]
                    if self.decomp[idx+2]==self.hosonto and self.decomp[idx+3]==_pair:
                        self.decomp[idx]=None
                        self.decomp[idx+1]=None
            
        except Exception as e:
            pass
    
  
    def __reconstructDecomp(self):
        '''
            reconstructs the word from decomp
        '''
        self.decomp=[x for x in self.decomp if x is not None] 
        self.word=''
        for ch in self.decomp:
            self.word+=ch 

    def __checkOp(self,op):
        '''
            execute an operation with  checking and None return
            args:
                opname : the function to execute
        '''
        # execute
        op()
        # reform
        self.decomp=[x for x in self.decomp if x is not None] 
        # check length
        if not self.__checkDecomp:
            self.return_none=True
        # return op success
        if self.return_none:
            return False
        else:
            return True
    
    def clean(self,word):
        '''
            cleans a given word
        '''
        if not isinstance(word, str):
            raise TypeError("The provided argument/ word is not a string") 
        
        
        self.word=word
        # None-flag
        self.return_none = False
        
        
        # replace Diacritics
        self.__replaceDiacritics()
        # create clean decomp
        self.__createDecomp()
        # check return
        if self.return_none:
            return None
        

        # list of operations
        ops=[self.__cleanInvalidEnds,
             self.__cleanInvalidStarts,
             self.__cleanNuktaUnicode,
             self.__cleanInvalidHosonto,
             self.__cleanInvalidToAndHosonto,
             self.__cleanDoubleVowelDiacritics,
             self.__cleanVowelDiacriticsComingAfterVowelsAndModifiers,
             self.__cleanInvalidMultipleConsonantDiacritics,
             self.__reconstructDecomp]
        

        for op in ops:
            if not self.__checkOp(op):
                return None
        
        return self.word
