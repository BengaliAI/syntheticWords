# -*-coding: utf-8 -
'''
    @author: Tahsin Reasat
    Adoptation:MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import pandas as pd

#--------------------
# Parser class
#--------------------
class GraphemeParser():
    def __init__(self,class_map_csv):
        # gets class map
        self.class_map_csv=class_map_csv
        # initializes components
        self.__getComps()
    
    def __getComps(self):
        '''
            **Private Initialization**

            reads and creates dataframe for roots,consonant_diacritic,vowel_diacritic and graphemes 
            args:
                class_map_csv        : path of classes.csv
            returns:
                tuple(df_root,df_vd,df_cd)
                df_root          :     dataframe for grapheme roots
                df_vd            :     dataframe for vowel_diacritic 
                df_cd            :     dataframe for consonant_diacritic
                
        '''
        # read class map
        df_map=pd.read_csv(self.class_map_csv)
        # get grapheme roots
        df_root = df_map.groupby('component_type').get_group('grapheme_root')
        df_root.index = df_root['label']
        df_root = df_root.drop(columns = ['label','component_type'])
        # get vowel_diacritic
        df_vd = df_map.groupby('component_type').get_group('vowel_diacritic')
        df_vd.index = df_vd['label']
        df_vd = df_vd.drop(columns = ['label','component_type'])
        # get consonant_diacritic
        df_cd = df_map.groupby('component_type').get_group('consonant_diacritic')
        df_cd.index = df_cd['label']
        df_cd = df_cd.drop(columns = ['label','component_type'])
        
        self.vds    =df_vd.component.tolist()
        self.cds    =df_cd.component.tolist()
        self.roots  =df_root.component.tolist()

        

    def word2grapheme(self,word):
        graphemes = []
        grapheme = ''
        i = 0
        while i < len(word):
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

    