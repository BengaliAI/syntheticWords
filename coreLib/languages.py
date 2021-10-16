# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
from re import L


class bangla:
    iden                   =    "bangla"
    vowels                 =    ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']
    consonants             =    ['ক', 'খ', 'গ', 'ঘ', 'ঙ', 
                                'চ', 'ছ','জ', 'ঝ', 'ঞ', 
                                'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 
                                'ত', 'থ', 'দ', 'ধ', 'ন', 
                                'প', 'ফ', 'ব', 'ভ', 'ম', 
                                'য', 'র', 'ল', 'শ', 'ষ', 
                                'স', 'হ','ড়', 'ঢ়', 'য়','ং', 'ঃ','ৎ']
    numbers                =    ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']
    punctuations           =    ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', 
                                 '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
                                 '{', '|', '}', '~', '।']
    # for grapheme decomp
    vowel_diacritics       =    ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']
    consonant_diacritics   =    ['ঁ', 'র্', 'র্য', '্য', '্র', '্র্য', 'র্্র']
    modifiers              =    []
    connector              =    '্'
    # special charecters
    special_charecters     =    [connector,'\u200d']
    # for synthetic data creation
    top_exts               =    ['ই', 'ঈ', 'উ', 'ঊ', 'ঐ','ঔ','ট', 'ঠ',' ি', 'ী', 'ৈ', 'ৌ','ঁ','র্']
    bot_exts               =    ['ু', 'ূ', 'ৃ',]
    # dict graphemes
    dict_graphemes         =    ["ঁ","ং","ঃ","অ","অ্যা","আ","আঁ","ই","ইঁ","ঈ","উ","উঁ","ঊ","ঋ","এ","এঁ","এ্যা","ঐ","ও","ঔ","ক",
                                "কঁ","কা","কাঁ","কি","কিঁ","কী","কীঁ","কু","কুঁ","কূ","কূঁ","কৃ","কৃঁ","কে","কেঁ","কৈ","কো","কোঁ","কৌ","কৌঁ",
                                "ক্ক","ক্কা","ক্কি","ক্কী","ক্কু","ক্কূ","ক্কে","ক্কো","ক্ট","ক্টা","ক্টি","ক্টী","ক্টু","ক্টূ","ক্টে","ক্টো","ক্ট্য","ক্ট্যা","ক্ট্যি","ক্ট্যী",
                                "ক্ট্যু","ক্ট্যূ","ক্ট্যে","ক্ট্যো","ক্ট্র","ক্ট্রো","ক্ত","ক্তা","ক্তি","ক্তী","ক্তু","ক্তূ","ক্তৃ","ক্তে","ক্তো","ক্ত্র","ক্ত্রা","ক্ত্রি","ক্ত্রী","ক্ত্রু",
                                "ক্ত্রূ","ক্ত্রে","ক্ত্রো","ক্ন","ক্না","ক্নি","ক্নী","ক্নু","ক্নে","ক্নো","ক্ব","ক্বা","ক্বি","ক্বী","ক্বে","ক্বো","ক্য","ক্যা","ক্যি","ক্যী",
                                "ক্যু","ক্যূ","ক্যে","ক্যো","ক্র","ক্রা","ক্রি","ক্রী","ক্রু","ক্রূ","ক্রে","ক্রো","ক্র্যা","ক্ল","ক্লা","ক্লি","ক্লী","ক্লু","ক্লূ","ক্লে",
                                "ক্লো","ক্ষ","ক্ষা","ক্ষি","ক্ষী","ক্ষু","ক্ষূ","ক্ষে","ক্ষো","ক্ষৌ","ক্ষ্ণ","ক্ষ্ণা","ক্ষ্ণি","ক্ষ্ণী","ক্ষ্ণু","ক্ষ্ণূ","ক্ষ্ণে","ক্ষ্ণো","ক্ষ্ণৌ","ক্ষ্ব",
                                "ক্ষ্বা","ক্ষ্বি","ক্ষ্বী","ক্ষ্বু","ক্ষ্বূ","ক্ষ্বে","ক্ষ্বো","ক্ষ্ম","ক্ষ্মা","ক্ষ্মি","ক্ষ্মী","ক্ষ্মু","ক্ষ্মে","ক্ষ্মো","ক্ষ্য","ক্ষ্যা","ক্ষ্যি","ক্ষ্যী","ক্ষ্যু","ক্ষ্যূ",
                                "ক্ষ্যে","ক্ষ্যো","ক্স","ক্সা","ক্সি","ক্সী","ক্সু","ক্সূ","ক্সে","ক্সো","খ","খঁ","খা","খাঁ","খি","খিঁ","খী","খীঁ","খু","খুঁ",
                                "খূ","খূঁ","খৃ","খৃঁ","খে","খেঁ","খৈ","খো","খোঁ","খৌ","খৌঁ","খ্য","খ্যা","খ্যি","খ্যু","খ্যূ","খ্যে","খ্যো","খ্র","খ্রা",
                                "খ্রি","খ্রী","খ্রু","খ্রূ","খ্রে","খ্রো","গ","গঁ","গা","গাঁ","গি","গিঁ","গী","গীঁ","গু","গুঁ","গূ","গূঁ","গৃ","গৃঁ",
                                "গে","গেঁ","গৈ","গো","গোঁ","গৌ","গৌঁ","গ্গ","গ্গা","গ্গি","গ্গী","গ্গু","গ্গূ","গ্গে","গ্গো","গ্দ","গ্দা","গ্দি","গ্দী","গ্দু",
                                "গ্দূ","গ্দে","গ্দো","গ্ধ","গ্ধা","গ্ধি","গ্ধী","গ্ধু","গ্ধূ","গ্ধে","গ্ধো","গ্ধ্য","গ্ধ্যা","গ্ধ্যি","গ্ধ্যী","গ্ধ্যু","গ্ধ্যূ","গ্ধ্যে","গ্ধ্যো","গ্ন",
                                "গ্না","গ্নি","গ্নী","গ্নু","গ্নূ","গ্নে","গ্নো","গ্ন্য","গ্ন্যা","গ্ন্যি","গ্ন্যী","গ্ন্যু","গ্ন্যূ","গ্ন্যে","গ্ন্যো","গ্ব","গ্বা","গ্বি","গ্বী","গ্বু",
                                "গ্বূ","গ্বে","গ্বো","গ্ম","গ্মা","গ্মি","গ্মী","গ্মু","গ্মূ","গ্মে","গ্মো","গ্য","গ্যা","গ্যি","গ্যী","গ্যু","গ্যে","গ্যো","গ্র","গ্রা",
                                "গ্রি","গ্রী","গ্রু","গ্রূ","গ্রে","গ্রো","গ্র্য","গ্র্যা","গ্র্যি","গ্র্যী","গ্র্যু","গ্র্যূ","গ্র্যে","গ্র্যো","গ্ল","গ্লা","গ্লি","গ্লী","গ্লু","গ্লূ",
                                "গ্লে","গ্লো","গ্ল্যা","ঘ","ঘঁ","ঘা","ঘাঁ","ঘি","ঘিঁ","ঘী","ঘীঁ","ঘু","ঘুঁ","ঘূ","ঘূঁ","ঘৃ","ঘৃঁ","ঘে","ঘেঁ","ঘৈ",
                                "ঘো","ঘোঁ","ঘৌ","ঘৌঁ","ঘ্ন","ঘ্না","ঘ্নি","ঘ্নী","ঘ্নু","ঘ্নূ","ঘ্নে","ঘ্নো","ঘ্য","ঘ্যা","ঘ্যি","ঘ্যী","ঘ্যু","ঘ্যে","ঘ্যো","ঘ্র",
                                "ঘ্রা","ঘ্রি","ঘ্রী","ঘ্রু","ঘ্রূ","ঘ্রে","ঘ্রো","ঙ","ঙঁ","ঙা","ঙি","ঙী","ঙু","ঙুঁ","ঙূ","ঙে","ঙেঁ","ঙো","ঙোঁ","ঙৌ",
                                "ঙ্ক","ঙ্কা","ঙ্কি","ঙ্কী","ঙ্কু","ঙ্কূ","ঙ্কৃ","ঙ্কে","ঙ্কো","ঙ্ক্তি","ঙ্ক্য","ঙ্ক্যা","ঙ্ক্যি","ঙ্ক্যী","ঙ্ক্যু","ঙ্ক্যূ","ঙ্ক্যে","ঙ্ক্যো","ঙ্ক্ষ","ঙ্ক্ষা",
                                "ঙ্ক্ষি","ঙ্ক্ষী","ঙ্ক্ষু","ঙ্ক্ষূ","ঙ্ক্ষে","ঙ্ক্ষো","ঙ্খ","ঙ্খা","ঙ্খি","ঙ্খী","ঙ্খু","ঙ্খূ","ঙ্খে","ঙ্খো","ঙ্গ","ঙ্গা","ঙ্গি","ঙ্গী","ঙ্গু","ঙ্গূ",
                                "ঙ্গে","ঙ্গো","ঙ্গ্য","ঙ্গ্যা","ঙ্গ্যি","ঙ্গ্যী","ঙ্গ্যু","ঙ্গ্যূ","ঙ্গ্যে","ঙ্গ্যো","ঙ্ঘ","ঙ্ঘা","ঙ্ঘি","ঙ্ঘী","ঙ্ঘু","ঙ্ঘূ","ঙ্ঘে","ঙ্ঘো","ঙ্ঘ্য","ঙ্ঘ্যা",
                                "ঙ্ঘ্যি","ঙ্ঘ্যী","ঙ্ঘ্যু","ঙ্ঘ্যূ","ঙ্ঘ্যে","ঙ্ঘ্যো","ঙ্ঘ্রা","ঙ্ঘ্রি","ঙ্ঘ্রী","ঙ্ঘ্রু","ঙ্ঘ্রূ","ঙ্ঘ্রে","ঙ্ঘ্রো","ঙ্ম","ঙ্মা","ঙ্মি","ঙ্মী","ঙ্মু","ঙ্মূ","ঙ্মে",
                                "ঙ্মো","চ","চঁ","চা","চাঁ","চি","চিঁ","চী","চীঁ","চু","চুঁ","চূ","চূঁ","চৃ","চৃঁ","চে","চেঁ","চৈ","চো","চোঁ",
                                "চৌ","চৌঁ","চ্চ","চ্চা","চ্চি","চ্চী","চ্চু","চ্চূ","চ্চে","চ্চো","চ্ছ","চ্ছা","চ্ছি","চ্ছী","চ্ছু","চ্ছূ","চ্ছৃ","চ্ছে","চ্ছো","চ্ছ্ব",
                                "চ্ছ্বা","চ্ছ্বি","চ্ছ্বী","চ্ছ্বু","চ্ছ্বূ","চ্ছ্বে","চ্ছ্বো","চ্ছ্র","চ্ছ্রা","চ্ছ্রি","চ্ছ্রী","চ্ছ্রু","চ্ছ্রূ","চ্ছ্রে","চ্ছ্রো","চ্ঞ","চ্ঞা","চ্ঞি","চ্ঞী","চ্ঞু",
                                "চ্ঞূ","চ্ঞে","চ্ঞো","চ্য","চ্যা","চ্যি","চ্যী","চ্যু","চ্যূ","চ্যে","চ্যো","ছ","ছঁ","ছা","ছাঁ","ছি","ছিঁ","ছী","ছীঁ","ছু",
                                "ছুঁ","ছূ","ছূঁ","ছৃ","ছৃঁ","ছে","ছেঁ","ছৈ","ছো","ছোঁ","ছৌ","ছৌঁ","ছ্য","ছ্যা","ছ্যি","ছ্যী","ছ্যু","ছ্যূ","ছ্যে","ছ্যো",
                                "জ","জঁ","জা","জি","জিঁ","জী","জীঁ","জু","জুঁ","জূ","জূঁ","জৃ","জৃঁ","জে","জেঁ","জৈ","জো","জোঁ","জৌ","জৌঁ",
                                "জ্জ","জ্জা","জ্জি","জ্জী","জ্জু","জ্জূ","জ্জে","জ্জো","জ্জ্ব","জ্জ্বা","জ্জ্বি","জ্জ্বী","জ্জ্বু","জ্জ্বূ","জ্জ্বে","জ্জ্বো","জ্ঝ","জ্ঝা","জ্ঝি","জ্ঝী",
                                "জ্ঝু","জ্ঝূ","জ্ঝে","জ্ঝো","জ্ঞ","জ্ঞা","জ্ঞি","জ্ঞী","জ্ঞু","জ্ঞূ","জ্ঞে","জ্ঞো","জ্ব","জ্বা","জ্বি","জ্বী","জ্বু","জ্বূ","জ্বে","জ্বো",
                                "জ্য","জ্যা","জ্যি","জ্যী","জ্যু","জ্যূ","জ্যে","জ্যৈ","জ্যো","জ্র","জ্রা","জ্রি","জ্রী","জ্রু","জ্রূ","জ্রে","জ্রো","ঝ","ঝঁ","ঝা",
                                "ঝাঁ","ঝি","ঝিঁ","ঝী","ঝীঁ","ঝু","ঝুঁ","ঝূ","ঝূঁ","ঝৃ","ঝৃঁ","ঝে","ঝেঁ","ঝৈ","ঝো","ঝোঁ","ঝৌ","ঝৌঁ","ঞ","ঞঁ",
                                "ঞা","ঞি","ঞী","ঞু","ঞুঁ","ঞূ","ঞে","ঞো","ঞৌ","ঞ্চ","ঞ্চা","ঞ্চি","ঞ্চী","ঞ্চু","ঞ্চূ","ঞ্চে","ঞ্চো","ঞ্ছ","ঞ্ছা","ঞ্ছি",
                                "ঞ্ছী","ঞ্ছু","ঞ্ছূ","ঞ্ছে","ঞ্ছো","ঞ্জ","ঞ্জা","ঞ্জি","ঞ্জী","ঞ্জু","ঞ্জূ","ঞ্জে","ঞ্জো","ঞ্ঝ","ঞ্ঝা","ঞ্ঝি","ঞ্ঝী","ঞ্ঝু","ঞ্ঝূ","ঞ্ঝে",
                                "ঞ্ঝো","ট","টঁ","টা","টি","টিঁ","টী","টীঁ","টু","টুঁ","টূ","টূঁ","টৃ","টৃঁ","টে","টেঁ","টৈ","টো","টোঁ","টৌ",
                                "টৌঁ","ট্ট","ট্টা","ট্টি","ট্টী","ট্টু","ট্টূ","ট্টে","ট্টো","ট্ব","ট্বা","ট্বি","ট্বী","ট্বু","ট্বূ","ট্বে","ট্বো","ট্ম","ট্মা","ট্মি",
                                "ট্মী","ট্মু","ট্মূ","ট্মে","ট্মো","ট্য","ট্যা","ট্যি","ট্যী","ট্যু","ট্যূ","ট্যে","ট্যো","ট্র","ট্রা","ট্রি","ট্রী","ট্রু","ট্রূ","ট্রে",
                                "ট্রো","ট্র্য","ট্র্যা","ট্র্যি","ট্র্যী","ট্র্যু","ট্র্যূ","ট্র্যে","ট্র্যো","ঠ","ঠঁ","ঠা","ঠাঁ","ঠি","ঠিঁ","ঠী","ঠীঁ","ঠু","ঠুঁ","ঠূ",
                                "ঠূঁ","ঠৃ","ঠৃঁ","ঠে","ঠেঁ","ঠৈ","ঠো","ঠোঁ","ঠৌ","ঠৌঁ","ঠ্য","ঠ্যা","ঠ্যি","ঠ্যী","ঠ্যু","ঠ্যূ","ঠ্যে","ঠ্যো","ড","ডঁ",
                                "ডা","ডি","ডিঁ","ডী","ডীঁ","ডু","ডুঁ","ডূ","ডূঁ","ডৃ","ডৃঁ","ডে","ডেঁ","ডৈ","ডো","ডোঁ","ডৌ","ডৌঁ","ড্ড","ড্ডা",
                                "ড্ডি","ড্ডী","ড্ডু","ড্ডূ","ড্ডে","ড্ডো","ড্ব","ড্বা","ড্বি","ড্বী","ড্বু","ড্বূ","ড্বে","ড্বো","ড্ম","ড্মা","ড্মি","ড্মী","ড্মু","ড্মূ",
                                "ড্মে","ড্মো","ড্য","ড্যা","ড্যি","ড্যী","ড্যু","ড্যূ","ড্যে","ড্যো","ড্র","ড্রা","ড্রি","ড্রী","ড্রু","ড্রূ","ড্রে","ড্রো","ঢ","ঢঁ",
                                "ঢা","ঢি","ঢিঁ","ঢী","ঢীঁ","ঢু","ঢুঁ","ঢূ","ঢূঁ","ঢৃ","ঢৃঁ","ঢে","ঢেঁ","ঢৈ","ঢো","ঢোঁ","ঢৌ","ঢৌঁ","ঢ্য","ঢ্যা",
                                "ঢ্যি","ঢ্যী","ঢ্যু","ঢ্যূ","ঢ্যে","ঢ্যো","ঢ্র","ঢ্রা","ঢ্রি","ঢ্রী","ঢ্রু","ঢ্রূ","ঢ্রে","ঢ্রো","ণ","ণঁ","ণা","ণি","ণিঁ","ণী",
                                "ণীঁ","ণু","ণুঁ","ণূ","ণূঁ","ণৃ","ণৃঁ","ণে","ণেঁ","ণৈ","ণো","ণোঁ","ণৌ","ণৌঁ","ণ্ট","ণ্টা","ণ্টি","ণ্টী","ণ্টু","ণ্টূ",
                                "ণ্টে","ণ্টো","ণ্ঠ","ণ্ঠা","ণ্ঠি","ণ্ঠী","ণ্ঠু","ণ্ঠূ","ণ্ঠে","ণ্ঠো","ণ্ঠ্য","ণ্ঠ্যা","ণ্ঠ্যি","ণ্ঠ্যী","ণ্ঠ্যু","ণ্ঠ্যূ","ণ্ঠ্যে","ণ্ঠ্যো","ণ্ড","ণ্ডা",
                                "ণ্ডি","ণ্ডী","ণ্ডু","ণ্ডূ","ণ্ডে","ণ্ডো","ণ্ড্য","ণ্ড্যা","ণ্ড্যি","ণ্ড্যী","ণ্ড্যু","ণ্ড্যূ","ণ্ড্যে","ণ্ড্যো","ণ্ড্র","ণ্ড্রা","ণ্ড্রি","ণ্ড্রী","ণ্ড্রু","ণ্ড্রূ",
                                "ণ্ড্রে","ণ্ড্রো","ণ্ঢ","ণ্ঢা","ণ্ঢি","ণ্ঢী","ণ্ঢু","ণ্ঢূ","ণ্ঢে","ণ্ঢো","ণ্ণ","ণ্ণা","ণ্ণি","ণ্ণী","ণ্ণু","ণ্ণূ","ণ্ণে","ণ্ণো","ণ্ব","ণ্বা",
                                "ণ্বি","ণ্বী","ণ্বু","ণ্বূ","ণ্বে","ণ্বো","ণ্ম","ণ্মা","ণ্মি","ণ্মী","ণ্মু","ণ্মূ","ণ্মে","ণ্মো","ণ্য","ণ্যা","ণ্যি","ণ্যী","ণ্যু","ণ্যূ",
                                "ণ্যে","ণ্যো","ত","তঁ","তা","তাঁ","তি","তিঁ","তী","তীঁ","তু","তুঁ","তূ","তৃ","তৃঁ","তে","তেঁ","তৈ","তো","তোঁ",
                                "তৌ","তৌঁ","ত্ত","ত্তা","ত্তি","ত্তী","ত্তু","ত্তূ","ত্তে","ত্তো","ত্ত্ব","ত্ত্বা","ত্ত্বি","ত্ত্বী","ত্ত্বূ","ত্ত্বে","ত্ত্বো","ত্ত্য","ত্থ","ত্থা",
                                "ত্থি","ত্থী","ত্থু","ত্থূ","ত্থে","ত্থো","ত্ন","ত্না","ত্নি","ত্নী","ত্নু","ত্নূ","ত্নে","ত্নো","ত্ব","ত্বা","ত্বি","ত্বী","ত্বু","ত্বূ",
                                "ত্বে","ত্বো","ত্ম","ত্মা","ত্মি","ত্মী","ত্মু","ত্মূ","ত্মে","ত্মো","ত্ম্য","ত্ম্যা","ত্ম্যি","ত্ম্যী","ত্ম্যু","ত্ম্যূ","ত্ম্যে","ত্ম্যো","ত্য","ত্যা",
                                "ত্যি","ত্যী","ত্যু","ত্যূ","ত্যে","ত্যো","ত্র","ত্রা","ত্রি","ত্রী","ত্রু","ত্রূ","ত্রে","ত্রৈ","ত্রো","ত্র্য","ত্র্যা","ত্র্যি","ত্র্যী","ত্র্যু",
                                "ত্র্যূ","ত্র্যে","ত্র্যো","থ","থঁ","থা","থি","থিঁ","থী","থীঁ","থু","থুঁ","থূ","থূঁ","থৃ","থৃঁ","থে","থেঁ","থৈ","থো",
                                "থোঁ","থৌ","থৌঁ","থ্ব","থ্বা","থ্বি","থ্বী","থ্বু","থ্বূ","থ্বে","থ্বো","থ্য","থ্যা","থ্যি","থ্যী","থ্যু","থ্যূ","থ্যে","থ্যো","থ্র",
                                "থ্রা","থ্রি","থ্রী","থ্রু","থ্রূ","থ্রে","থ্রো","দ","দঁ","দা","দাঁ","দি","দিঁ","দী","দীঁ","দু","দুঁ","দূ","দূঁ","দূ্র্য",
                                "দৃ","দৃঁ","দে","দেঁ","দৈ","দো","দোঁ","দৌ","দৌঁ","দ্গ","দ্গা","দ্গি","দ্গী","দ্গু","দ্গূ","দ্গে","দ্গো","দ্ঘ","দ্ঘা","দ্ঘি",
                                "দ্ঘী","দ্ঘু","দ্ঘূ","দ্ঘে","দ্ঘো","দ্দ","দ্দা","দ্দি","দ্দী","দ্দু","দ্দূ","দ্দে","দ্দো","দ্দৌ","দ্ধ","দ্ধা","দ্ধি","দ্ধী","দ্ধু","দ্ধূ",
                                "দ্ধৃ","দ্ধে","দ্ধো","দ্ব","দ্বা","দ্বি","দ্বী","দ্বু","দ্বূ","দ্বৃ","দ্বে","দ্বৈ","দ্বো","দ্ব্য","দ্ব্যা","দ্ব্যি","দ্ব্যী","দ্ব্যু","দ্ব্যূ","দ্ব্যে",
                                "দ্ব্যো","দ্ভ","দ্ভা","দ্ভি","দ্ভী","দ্ভু","দ্ভূ","দ্ভে","দ্ভো","দ্ভ্রা","দ্ম","দ্মা","দ্মি","দ্মী","দ্মু","দ্মূ","দ্মে","দ্মো","দ্য","দ্যা",
                                "দ্যি","দ্যী","দ্যু","দ্যূ","দ্যে","দ্যো","দ্র","দ্রা","দ্রি","দ্রী","দ্রু","দ্রূ","দ্রে","দ্রো","দ্রৌ","দ্র্য","দ্র্যা","দ্র্যি","দ্র্যী","দ্র্যু",
                                "দ্র্যে","দ্র্যো","ধ","ধঁ","ধা","ধাঁ","ধি","ধিঁ","ধী","ধীঁ","ধু","ধুঁ","ধূ","ধূঁ","ধৃ","ধৃঁ","ধে","ধেঁ","ধৈ","ধো",
                                "ধোঁ","ধৌ","ধৌঁ","ধ্ন","ধ্না","ধ্নি","ধ্নী","ধ্নু","ধ্নূ","ধ্নে","ধ্নো","ধ্ব","ধ্বা","ধ্বি","ধ্বী","ধ্বু","ধ্বূ","ধ্বে","ধ্বো","ধ্ম",
                                "ধ্মা","ধ্মি","ধ্মী","ধ্মু","ধ্মূ","ধ্মে","ধ্মো","ধ্য","ধ্যা","ধ্যি","ধ্যী","ধ্যু","ধ্যূ","ধ্যে","ধ্যো","ধ্র","ধ্রা","ধ্রি","ধ্রী","ধ্রু",
                                "ধ্রূ","ধ্রে","ধ্রো","ন","নঁ","না","নি","নিঁ","নী","নীঁ","নু","নুঁ","নূ","নূঁ","নৃ","নৃঁ","নে","নেঁ","নৈ","নো",
                                "নোঁ","নৌ","নৌঁ","ন্জী","ন্জু","ন্ট","ন্টা","ন্টি","ন্টী","ন্টু","ন্টূ","ন্টে","ন্টো","ন্ট্রা","ন্ট্রি","ন্ট্রো","ন্ঠ","ন্ঠা","ন্ঠি","ন্ঠী",
                                "ন্ঠু","ন্ঠূ","ন্ঠে","ন্ঠো","ন্ড","ন্ডা","ন্ডি","ন্ডী","ন্ডু","ন্ডূ","ন্ডে","ন্ডো","ন্ড্র","ন্ড্রা","ন্ড্রি","ন্ড্রী","ন্ড্রু","ন্ড্রূ","ন্ড্রে","ন্ড্রো",
                                "ন্ত","ন্তা","ন্তি","ন্তী","ন্তু","ন্তূ","ন্তে","ন্তো","ন্ত্ব","ন্ত্বা","ন্ত্বি","ন্ত্বী","ন্ত্বু","ন্ত্বূ","ন্ত্বে","ন্ত্বো","ন্ত্য","ন্ত্যা","ন্ত্যি","ন্ত্যী",
                                "ন্ত্যু","ন্ত্যূ","ন্ত্যে","ন্ত্যো","ন্ত্র","ন্ত্রা","ন্ত্রি","ন্ত্রী","ন্ত্রু","ন্ত্রূ","ন্ত্রে","ন্ত্রো","ন্ত্র্য","ন্ত্র্যা","ন্ত্র্যি","ন্ত্র্যী","ন্ত্র্যু","ন্ত্র্যূ","ন্ত্র্যে","ন্ত্র্যো",
                                "ন্থ","ন্থা","ন্থি","ন্থী","ন্থে","ন্দ","ন্দা","ন্দি","ন্দী","ন্দু","ন্দূ","ন্দে","ন্দো","ন্দ্ব","ন্দ্বা","ন্দ্বি","ন্দ্বী","ন্দ্বু","ন্দ্বূ","ন্দ্বে",
                                "ন্দ্বো","ন্দ্য","ন্দ্যা","ন্দ্যি","ন্দ্যী","ন্দ্যু","ন্দ্যূ","ন্দ্যে","ন্দ্যো","ন্দ্র","ন্দ্রা","ন্দ্রি","ন্দ্রী","ন্দ্রু","ন্দ্রূ","ন্দ্রে","ন্দ্রো","ন্ধ","ন্ধা","ন্ধি",
                                "ন্ধী","ন্ধু","ন্ধূ","ন্ধূ্র","ন্ধে","ন্ধো","ন্ধ্য","ন্ধ্যা","ন্ধ্যি","ন্ধ্যী","ন্ধ্যু","ন্ধ্যূ","ন্ধ্যে","ন্ধ্যো","ন্ধ্র","ন্ধ্রা","ন্ধ্রি","ন্ধ্রী","ন্ধ্রু","ন্ধ্রে",
                                "ন্ধ্রো","ন্ন","ন্না","ন্নি","ন্নী","ন্নু","ন্নূ","ন্নে","ন্নো","ন্ন্য","ন্ন্যা","ন্ন্যি","ন্ন্যী","ন্ন্যু","ন্ন্যূ","ন্ন্যে","ন্ন্যো","ন্ব","ন্বা","ন্বি",
                                "ন্বী","ন্বু","ন্বূ","ন্বে","ন্বো","ন্ম","ন্মা","ন্মি","ন্মী","ন্মু","ন্মূ","ন্মে","ন্মো","ন্য","ন্যা","ন্যি","ন্যী","ন্যু","ন্যূ","ন্যে",
                                "ন্যো","ন্স","ন্সা","ন্সি","ন্সী","ন্সু","ন্সূ","ন্সে","ন্সো","ন্হ","ন্হা","ন্হি","ন্হী","ন্হু","ন্হূ","ন্হে","ন্হো","প","পঁ","পা",
                                "পাঁ","পি","পিঁ","পী","পীঁ","পু","পুঁ","পূ","পূঁ","পৃ","পৃঁ","পে","পেঁ","পৈ","পো","পোঁ","পৌ","পৌঁ","প্ট","প্টা",
                                "প্টি","প্টী","প্টু","প্টূ","প্টে","প্টো","প্ত","প্তা","প্তি","প্তী","প্তু","প্তূ","প্তে","প্তো","প্ন","প্না","প্নি","প্নী","প্নু","প্নূ",
                                "প্নে","প্নো","প্প","প্পা","প্পি","প্পী","প্পু","প্পূ","প্পে","প্পো","প্য","প্যা","প্যি","প্যী","প্যু","প্যূ","প্যে","প্যো","প্র","প্রা",
                                "প্রি","প্রী","প্রু","প্রূ","প্রে","প্রো","প্র্য","প্র্যা","প্র্যি","প্র্যী","প্র্যু","প্র্যূ","প্র্যে","প্র্যো","প্ল","প্লা","প্লি","প্লী","প্লু","প্লূ",
                                "প্লে","প্লো","প্ল্য","প্ল্যা","প্ল্যি","প্ল্যী","প্ল্যু","প্ল্যূ","প্ল্যে","প্ল্যো","প্স","প্সা","প্সি","প্সী","প্সু","প্সূ","প্সে","প্সো","ফ","ফঁ",
                                "ফা","ফাঁ","ফি","ফিঁ","ফী","ফীঁ","ফু","ফুঁ","ফূ","ফৃ","ফৃঁ","ফে","ফেঁ","ফৈ","ফো","ফোঁ","ফৌ","ফৌঁ","ফ্ট","ফ্ফ",
                                "ফ্যা","ফ্র","ফ্রা","ফ্রি","ফ্রী","ফ্রু","ফ্রূ","ফ্রে","ফ্রো","ফ্র্যা","ফ্ল","ফ্লা","ফ্লি","ফ্লী","ফ্লু","ফ্লূ","ফ্লে","ফ্লো","ফ্ল্য","ফ্ল্যা",
                                "ফ্ল্যি","ফ্ল্যী","ফ্ল্যু","ফ্ল্যূ","ফ্ল্যে","ফ্ল্যো","ব","বঁ","বা","বাঁ","বি","বিঁ","বী","বীঁ","বু","বুঁ","বূ","বূঁ","বৃ","বৃঁ",
                                "বে","বেঁ","বৈ","বো","বোঁ","বৌ","বৌঁ","ব্জ","ব্জা","ব্জি","ব্জী","ব্জু","ব্জূ","ব্জে","ব্জো","ব্দ","ব্দা","ব্দি","ব্দী","ব্দু",
                                "ব্দূ","ব্দে","ব্দো","ব্ধ","ব্ধা","ব্ধি","ব্ধী","ব্ধু","ব্ধূ","ব্ধে","ব্ধো","ব্ব","ব্বা","ব্বি","ব্বী","ব্বু","ব্বূ","ব্বে","ব্বো","ব্য",
                                "ব্যা","ব্যি","ব্যী","ব্যু","ব্যূ","ব্যে","ব্যো","ব্র","ব্রা","ব্রি","ব্রী","ব্রু","ব্রূ","ব্রে","ব্রো","ব্র্যা","ব্ল","ব্লা","ব্লি","ব্লী",
                                "ব্লু","ব্লূ","ব্লে","ব্লো","ব্ল্য","ব্ল্যা","ব্ল্যি","ব্ল্যী","ব্ল্যু","ব্ল্যূ","ব্ল্যে","ব্ল্যো","ভ","ভঁ","ভা","ভাঁ","ভি","ভিঁ","ভী","ভীঁ",
                                "ভু","ভুঁ","ভূ","ভূঁ","ভৃ","ভৃঁ","ভে","ভেঁ","ভৈ","ভো","ভোঁ","ভৌ","ভৌঁ","ভ্য","ভ্যা","ভ্যি","ভ্যী","ভ্যু","ভ্যূ","ভ্যে",
                                "ভ্যো","ভ্র","ভ্রা","ভ্রি","ভ্রী","ভ্রু","ভ্রূ","ভ্রে","ভ্রো","ভ্ল","ভ্লা","ভ্লি","ভ্লী","ভ্লু","ভ্লূ","ভ্লে","ভ্লো","ম","মঁ","মা",
                                "মি","মিঁ","মী","মীঁ","মু","মুঁ","মূ","মূঁ","মূ্ন","মৃ","মৃঁ","মে","মেঁ","মৈ","মো","মোঁ","মৌ","মৌঁ","ম্ন","ম্না",
                                "ম্নি","ম্নী","ম্নু","ম্নে","ম্নো","ম্প","ম্পা","ম্পি","ম্পী","ম্পু","ম্পূ","ম্পৃ","ম্পে","ম্পো","ম্প্র","ম্প্রা","ম্প্রি","ম্প্রী","ম্প্রু","ম্প্রূ",
                                "ম্প্রে","ম্প্রো","ম্ফ","ম্ফা","ম্ফি","ম্ফী","ম্ফু","ম্ফূ","ম্ফে","ম্ফো","ম্ব","ম্বা","ম্বি","ম্বী","ম্বু","ম্বূ","ম্বে","ম্বো","ম্ভ","ম্ভা",
                                "ম্ভি","ম্ভী","ম্ভু","ম্ভূ","ম্ভে","ম্ভো","ম্ভ্র","ম্ভ্রা","ম্ভ্রি","ম্ভ্রী","ম্ভ্রু","ম্ভ্রূ","ম্ভ্রে","ম্ভ্রো","ম্ম","ম্মা","ম্মি","ম্মী","ম্মু","ম্মূ",
                                "ম্মে","ম্মো","ম্য","ম্যা","ম্যি","ম্যী","ম্যু","ম্যূ","ম্যে","ম্যো","ম্র","ম্রা","ম্রি","ম্রী","ম্রু","ম্রূ","ম্রে","ম্রো","ম্ল","ম্লা",
                                "ম্লি","ম্লী","ম্লু","ম্লূ","ম্লে","ম্লো","য","যঁ","যা","যি","যিঁ","যী","যীঁ","যু","যুঁ","যূ","যূঁ","যৃ","যৃঁ","যে",
                                "যেঁ","যৈ","যো","যোঁ","যৌ","যৌঁ","য্য","য্যা","য্যি","য্যী","য্যু","য্যূ","য্যে","য্যো","র","রঁ","রা","রাঁ","রি","রিঁ",
                                "রী","রীঁ","রু","রুঁ","রূ","রূঁ","রৃ","রৃঁ","রে","রেঁ","রৈ","রো","রোঁ","রৌ","রৌঁ","র্ক","র্কা","র্কি","র্কী","র্কু",
                                "র্কূ","র্কে","র্কো","র্ক্য","র্ক্যা","র্ক্যু","র্ক্যে","র্ক্যো","র্ক্স","র্ক্সী","র্খ","র্খা","র্খি","র্খী","র্খু","র্খূ","র্খে","র্খো","র্খ্য","র্খ্যা",
                                "র্খ্যু","র্খ্যে","র্খ্যো","র্গ","র্গা","র্গি","র্গী","র্গু","র্গূ","র্গে","র্গো","র্গ্য","র্গ্যা","র্গ্যু","র্গ্যূ","র্গ্যে","র্গ্যো","র্ঘ","র্ঘা","র্ঘি",
                                "র্ঘী","র্ঘু","র্ঘূ","র্ঘে","র্ঘো","র্ঘ্য","র্ঘ্যা","র্ঘ্যু","র্ঘ্যূ","র্ঘ্যে","র্ঘ্যো","র্চ","র্চা","র্চি","র্চী","র্চু","র্চূ","র্চে","র্চো","র্চ্য",
                                "র্চ্যা","র্চ্যু","র্চ্যূ","র্চ্যে","র্চ্যো","র্ছ","র্ছা","র্ছি","র্ছী","র্ছু","র্ছূ","র্ছে","র্ছো","র্ছ্য","র্ছ্যা","র্ছ্যু","র্ছ্যূ","র্ছ্যে","র্ছ্যো","র্জ",
                                "র্জা","র্জি","র্জী","র্জু","র্জূ","র্জে","র্জো","র্জ্য","র্জ্যা","র্জ্যু","র্জ্যূ","র্জ্যে","র্জ্যো","র্ঝ","র্ঝা","র্ঝি","র্ঝী","র্ঝু","র্ঝূ","র্ঝে",
                                "র্ঝো","র্ঝ্য","র্ঝ্যা","র্ঝ্যু","র্ঝ্যূ","র্ঝ্যে","র্ঝ্যো","র্ট","র্টা","র্টি","র্টী","র্টু","র্টূ","র্টে","র্টো","র্ট্য","র্ট্যা","র্ট্যু","র্ট্যূ","র্ট্যে",
                                "র্ট্যো","র্ঠ","র্ঠা","র্ঠি","র্ঠী","র্ঠু","র্ঠূ","র্ঠে","র্ঠো","র্ঠ্য","র্ঠ্যা","র্ঠ্যু","র্ঠ্যূ","র্ঠ্যে","র্ঠ্যো","র্ড","র্ডা","র্ডি","র্ডী","র্ডু",
                                "র্ডূ","র্ডে","র্ডো","র্ড্য","র্ড্যা","র্ড্যু","র্ড্যূ","র্ড্যে","র্ড্যো","র্ঢ","র্ঢা","র্ঢি","র্ঢী","র্ঢু","র্ঢূ","র্ঢে","র্ঢো","র্ঢ্য","র্ঢ্যা","র্ঢ্যু",
                                "র্ঢ্যূ","র্ঢ্যে","র্ঢ্যো","র্ণ","র্ণা","র্ণি","র্ণী","র্ণু","র্ণূ","র্ণে","র্ণো","র্ণ্য","র্ণ্যা","র্ণ্যু","র্ণ্যূ","র্ণ্যে","র্ণ্যো","র্ত","র্তা","র্তি",
                                "র্তী","র্তু","র্তূ","র্তৃ","র্তে","র্তো","র্ত্ত","র্ত্তী","র্ত্য","র্ত্যা","র্ত্যু","র্ত্যূ","র্ত্যে","র্ত্যো","র্ত্রী","র্ত্রে","র্থ","র্থা","র্থি","র্থী",
                                "র্থু","র্থূ","র্থে","র্থো","র্থ্য","র্থ্যা","র্থ্যু","র্থ্যূ","র্থ্যে","র্থ্যো","র্দ","র্দা","র্দি","র্দী","র্দু","র্দূ","র্দৃ","র্দে","র্দো","র্দ্দ",
                                "র্দ্দা","র্দ্ধ","র্দ্ধে","র্দ্ব","র্দ্বি","র্দ্য","র্দ্যা","র্দ্যু","র্দ্যূ","র্দ্যে","র্দ্যো","র্দ্র","র্ধ","র্ধা","র্ধি","র্ধী","র্ধু","র্ধূ","র্ধে","র্ধো",
                                "র্ধ্ব","র্ধ্বে","র্ধ্য","র্ধ্যা","র্ধ্যু","র্ধ্যূ","র্ধ্যে","র্ধ্যো","র্ন","র্না","র্নি","র্নী","র্নু","র্নূ","র্নে","র্নো","র্ন্ত","র্ন্য","র্ন্যা","র্ন্যু",
                                "র্ন্যূ","র্ন্যে","র্ন্যো","র্প","র্পা","র্পি","র্পী","র্পু","র্পূ","র্পে","র্পো","র্প্য","র্প্যা","র্প্যু","র্প্যূ","র্প্যে","র্প্যো","র্ফ","র্ফা","র্ফি",
                                "র্ফী","র্ফু","র্ফূ","র্ফে","র্ফো","র্ফ্য","র্ফ্যা","র্ফ্যু","র্ফ্যূ","র্ফ্যে","র্ফ্যো","র্ব","র্বা","র্বি","র্বী","র্বু","র্বূ","র্বৃ","র্বে","র্বো",
                                "র্ব্ব","র্ব্য","র্ব্যা","র্ব্যু","র্ব্যূ","র্ব্যে","র্ব্যো","র্ভ","র্ভা","র্ভি","র্ভী","র্ভু","র্ভূ","র্ভে","র্ভো","র্ভ্য","র্ভ্যা","র্ভ্যু","র্ভ্যূ","র্ভ্যে",
                                "র্ভ্যো","র্ম","র্মা","র্মি","র্মী","র্মু","র্মূ","র্মে","র্মো","র্ম্ম","র্ম্য","র্ম্যা","র্ম্যু","র্ম্যূ","র্ম্যে","র্ম্যো","র্য","র্যা","র্যি","র্যী",
                                "র্যু","র্যূ","র্যে","র্যো","র্য্য","র্য্যা","র্য্যু","র্য্যূ","র্য্যে","র্য্যো","র্র","র্রা","র্রি","র্রী","র্রু","র্রূ","র্রে","র্রো","র্র্য","র্র্যা",
                                "র্র্যু","র্র্যূ","র্র্যে","র্র্যো","র্ল","র্লা","র্লি","র্লী","র্লু","র্লূ","র্লে","র্লো","র্ল্ড","র্ল্য","র্ল্যা","র্ল্যু","র্ল্যূ","র্ল্যে","র্ল্যো","র্শ",
                                "র্শা","র্শি","র্শী","র্শু","র্শূ","র্শে","র্শো","র্শ্ব","র্শ্বি","র্শ্বে","র্শ্য","র্শ্যা","র্শ্যু","র্শ্যূ","র্শ্যে","র্শ্যো","র্ষ","র্ষা","র্ষি","র্ষী",
                                "র্ষু","র্ষূ","র্ষে","র্ষো","র্ষ্য","র্ষ্যা","র্ষ্যু","র্ষ্যূ","র্ষ্যে","র্ষ্যো","র্স","র্সা","র্সি","র্সী","র্সু","র্সূ","র্সে","র্সো","র্স্ট","র্স্থা",
                                "র্স্য","র্স্যা","র্স্যু","র্স্যূ","র্স্যে","র্স্যো","র্হ","র্হা","র্হি","র্হী","র্হু","র্হূ","র্হে","র্হো","র্হ্য","র্হ্যা","র্হ্যু","র্হ্যূ","র্হ্যে","র্হ্যো",
                                "র্ড়","র্ড়া","র্ড়ি","র্ড়ী","র্ড়ু","র্ড়ূ","র্ড়ে","র্ড়ো","র্ড়্য","র্ড়্যা","র্ড়্যু","র্ড়্যূ","র্ড়্যে","র্ড়্যো","র্ঢ়","র্ঢ়া","র্ঢ়ি","র্ঢ়ী","র্ঢ়ু","র্ঢ়ূ",
                                "র্ঢ়ে","র্ঢ়ো","র্ঢ়্য","র্ঢ়্যা","র্ঢ়্যু","র্ঢ়্যূ","র্ঢ়্যে","র্ঢ়্যো","র্য়","র্য়া","র্য়ি","র্য়ী","র্য়ু","র্য়ূ","র্য়ে","র্য়ো","র্য়্য","র্য়্যা","র্য়্যু","র্য়্যূ",
                                "র্য়্যে","র্য়্যো","র‍্যা","ল","লঁ","লা","লি","লিঁ","লী","লীঁ","লু","লুঁ","লূ","লূঁ","লৃ","লৃঁ","লে","লেঁ","লৈ","লো",
                                "লোঁ","লৌ","লৌঁ","ল্ক","ল্কা","ল্কি","ল্কী","ল্কু","ল্কূ","ল্কে","ল্কো","ল্ক্য","ল্ক্যা","ল্ক্যি","ল্ক্যী","ল্ক্যু","ল্ক্যূ","ল্ক্যে","ল্ক্যো","ল্গ",
                                "ল্গা","ল্গি","ল্গী","ল্গু","ল্গূ","ল্গে","ল্গো","ল্ট","ল্টা","ল্টি","ল্টী","ল্টু","ল্টূ","ল্টে","ল্টো","ল্ড","ল্ডা","ল্ডি","ল্ডী","ল্ডু",
                                "ল্ডূ","ল্ডে","ল্ডো","ল্প","ল্পা","ল্পি","ল্পী","ল্পু","ল্পূ","ল্পে","ল্পো","ল্ফ","ল্ফা","ল্ফি","ল্ফী","ল্ফু","ল্ফূ","ল্ফে","ল্ফো","ল্ব",
                                "ল্বা","ল্বি","ল্বী","ল্বু","ল্বূ","ল্বে","ল্বো","ল্ভ","ল্ভা","ল্ভি","ল্ভী","ল্ভু","ল্ভূ","ল্ভে","ল্ভো","ল্ম","ল্মা","ল্মি","ল্মী","ল্মু",
                                "ল্মূ","ল্মে","ল্মো","ল্য","ল্যা","ল্যি","ল্যী","ল্যু","ল্যূ","ল্যে","ল্যো","ল্ল","ল্লা","ল্লি","ল্লী","ল্লু","ল্লূ","ল্লে","ল্লো","শ",
                                "শঁ","শা","শি","শিঁ","শী","শীঁ","শু","শুঁ","শূ","শূঁ","শৃ","শৃঁ","শে","শেঁ","শৈ","শো","শোঁ","শৌ","শৌঁ","শ্চ",
                                "শ্চা","শ্চি","শ্চী","শ্চু","শ্চূ","শ্চে","শ্চো","শ্ছ","শ্ছা","শ্ছি","শ্ছী","শ্ছু","শ্ছূ","শ্ছে","শ্ছো","শ্ন","শ্না","শ্নি","শ্নী","শ্নু",
                                "শ্নূ","শ্নে","শ্নো","শ্ব","শ্বা","শ্বি","শ্বী","শ্বু","শ্বূ","শ্বে","শ্বো","শ্ম","শ্মা","শ্মি","শ্মী","শ্মু","শ্মূ","শ্মে","শ্মো","শ্য",
                                "শ্যা","শ্যি","শ্যী","শ্যু","শ্যূ","শ্যে","শ্যো","শ্র","শ্রা","শ্রি","শ্রী","শ্রু","শ্রূ","শ্রে","শ্রো","শ্ল","শ্লা","শ্লি","শ্লী","শ্লু",
                                "শ্লে","শ্লো","ষ","ষঁ","ষা","ষি","ষিঁ","ষী","ষীঁ","ষু","ষুঁ","ষূ","ষূঁ","ষৃ","ষৃঁ","ষে","ষেঁ","ষৈ","ষো","ষোঁ",
                                "ষৌ","ষৌঁ","ষ্ক","ষ্কা","ষ্কি","ষ্কী","ষ্কু","ষ্কূ","ষ্কৃ","ষ্কে","ষ্কো","ষ্ক্র","ষ্ক্রা","ষ্ক্রি","ষ্ক্রী","ষ্ক্রু","ষ্ক্রে","ষ্ক্রো","ষ্ট","ষ্টা",
                                "ষ্টি","ষ্টী","ষ্টু","ষ্টূ","ষ্টূ্র","ষ্টে","ষ্টো","ষ্ট্য","ষ্ট্যা","ষ্ট্যি","ষ্ট্যী","ষ্ট্যু","ষ্ট্যূ","ষ্ট্যে","ষ্ট্যো","ষ্ট্র","ষ্ট্রা","ষ্ট্রি","ষ্ট্রী","ষ্ট্রু",
                                "ষ্ট্রে","ষ্ট্রো","ষ্ঠ","ষ্ঠা","ষ্ঠি","ষ্ঠী","ষ্ঠু","ষ্ঠূ","ষ্ঠে","ষ্ঠো","ষ্ঠ্য","ষ্ঠ্যা","ষ্ঠ্যি","ষ্ঠ্যী","ষ্ঠ্যু","ষ্ঠ্যূ","ষ্ঠ্যে","ষ্ঠ্যো","ষ্ণ","ষ্ণা",
                                "ষ্ণি","ষ্ণী","ষ্ণু","ষ্ণূ","ষ্ণে","ষ্ণো","ষ্ণ্য","ষ্ণ্যা","ষ্ণ্যি","ষ্ণ্যী","ষ্ণ্যু","ষ্ণ্যূ","ষ্ণ্যে","ষ্ণ্যো","ষ্ণ্হ","ষ্প","ষ্পা","ষ্পি","ষ্পী","ষ্পু",
                                "ষ্পূ","ষ্পে","ষ্পো","ষ্প্র","ষ্প্রা","ষ্প্রি","ষ্প্রী","ষ্প্রু","ষ্প্রূ","ষ্প্রে","ষ্প্রো","ষ্ফ","ষ্ফা","ষ্ফি","ষ্ফী","ষ্ফু","ষ্ফূ","ষ্ফে","ষ্ফো","ষ্ব",
                                "ষ্বা","ষ্বি","ষ্বী","ষ্বু","ষ্বূ","ষ্বে","ষ্বো","ষ্ম","ষ্মা","ষ্মি","ষ্মী","ষ্মু","ষ্মূ","ষ্মে","ষ্মো","ষ্ম্য","ষ্ম্যা","ষ্ম্যি","ষ্ম্যী","ষ্ম্যু",
                                "ষ্ম্যূ","ষ্ম্যে","ষ্ম্যো","ষ্য","ষ্যা","ষ্যি","ষ্যী","ষ্যু","ষ্যূ","ষ্যে","ষ্যো","ষ্ূক্র","স","সঁ","সা","সাঁ","সি","সিঁ","সী","সীঁ",
                                "সু","সুঁ","সূ","সূঁ","সৃ","সৃঁ","সে","সেঁ","সৈ","সো","সোঁ","সৌ","সৌঁ","স্ক","স্কা","স্কি","স্কী","স্কু","স্কূ","স্কৃ",
                                "স্কে","স্কো","স্ক্যা","স্ক্র","স্ক্রা","স্ক্রি","স্ক্রী","স্ক্রু","স্ক্রূ","স্ক্রে","স্ক্রো","স্খ","স্খা","স্খি","স্খী","স্খু","স্খূ","স্খে","স্খো","স্ট",
                                "স্টা","স্টি","স্টী","স্টু","স্টূ","স্টে","স্টো","স্ট্য","স্ট্যা","স্ট্যি","স্ট্যী","স্ট্যু","স্ট্যূ","স্ট্যে","স্ট্যো","স্ট্র","স্ট্রা","স্ট্রি","স্ট্রী","স্ট্রু",
                                "স্ট্রূ","স্ট্রে","স্ট্রো","স্ট্র্যা","স্ত","স্তা","স্তি","স্তী","স্তু","স্তূ","স্তৃ","স্তে","স্তো","স্ত্য","স্ত্যা","স্ত্যি","স্ত্যী","স্ত্যু","স্ত্যূ","স্ত্যে",
                                "স্ত্যো","স্ত্র","স্ত্রা","স্ত্রি","স্ত্রী","স্ত্রু","স্ত্রূ","স্ত্রে","স্ত্রো","স্থ","স্থা","স্থি","স্থূ","স্থে","স্থ্য","স্থ্যে","স্ন","স্না","স্নি","স্নী",
                                "স্নু","স্নূ","স্নে","স্নো","স্প","স্পা","স্পি","স্পী","স্পু","স্পূ","স্পৃ","স্পে","স্পো","স্প্যা","স্প্রা","স্প্রি","স্ফ","স্ফা","স্ফি","স্ফী",
                                "স্ফু","স্ফূ","স্ফে","স্ফো","স্ব","স্বা","স্বি","স্বী","স্বু","স্বূ","স্বে","স্বৈ","স্বো","স্ম","স্মা","স্মি","স্মী","স্মু","স্মূ","স্মৃ",
                                "স্মে","স্মো","স্য","স্যা","স্যি","স্যী","স্যু","স্যূ","স্যে","স্যো","স্র","স্রা","স্রি","স্রী","স্রু","স্রূ","স্রে","স্রো","স্ল","স্লা",
                                "স্লি","স্লী","স্লূ","স্লে","স্লো","স্সু","স্হ","স্হা","স্হি","স্হী","স্হু","স্হূ","স্হে","স্হো","স্হ্য","স্হ্যা","স্হ্যি","স্হ্যী","স্হ্যু","স্হ্যূ",
                                "স্হ্যে","স্হ্যো","হ","হঁ","হা","হাঁ","হি","হিঁ","হী","হীঁ","হু","হুঁ","হূ","হূঁ","হৃ","হৃঁ","হে","হেঁ","হৈ","হো",
                                "হোঁ","হৌ","হৌঁ","হ্ণ","হ্ণা","হ্ণি","হ্ণী","হ্ণু","হ্ণূ","হ্ণে","হ্ণো","হ্ন","হ্না","হ্নি","হ্নী","হ্নু","হ্নূ","হ্নে","হ্নো","হ্ব",
                                "হ্বা","হ্বি","হ্বী","হ্বু","হ্বূ","হ্বে","হ্বো","হ্ম","হ্মা","হ্মি","হ্মী","হ্মু","হ্মূ","হ্মে","হ্মো","হ্য","হ্যা","হ্যি","হ্যী","হ্যু",
                                "হ্যূ","হ্যে","হ্যো","হ্র","হ্রা","হ্রি","হ্রী","হ্রু","হ্রূ","হ্রে","হ্রো","হ্ল","হ্লা","হ্লি","হ্লী","হ্লু","হ্লূ","হ্লে","হ্লো","ৎ",
                                "ড়","ড়ঁ","ড়া","ড়ি","ড়িঁ","ড়ী","ড়ীঁ","ড়ু","ড়ুঁ","ড়ূ","ড়ূঁ","ড়ৃ","ড়ে","ড়েঁ","ড়ৈ","ড়ো","ড়োঁ","ড়ৌ","ড়ৌঁ","ড়্গ",
                                "ড়্গা","ড়্গি","ড়্গী","ড়্গু","ড়্গূ","ড়্গে","ড়্গো","ঢ়","ঢ়ঁ","ঢ়া","ঢ়ি","ঢ়িঁ","ঢ়ী","ঢ়ীঁ","ঢ়ু","ঢ়ুঁ","ঢ়ূ","ঢ়ূঁ","ঢ়ৃ","ঢ়ে",
                                "ঢ়েঁ","ঢ়ৈ","ঢ়ো","ঢ়োঁ","ঢ়ৌ","ঢ়ৌঁ","য়","য়ঁ","য়া","য়ি","য়িঁ","য়ী","য়ীঁ","য়ু","য়ুঁ","য়ূ","য়ূঁ","য়ৃ","য়ে","য়েঁ",
                                "য়ৈ","য়ো","য়োঁ","য়ৌ","য়ৌঁ","য়্যা"] 
    roots                    =  ["ং","ঃ","অ","আ","ই","ঈ","উ","ঊ","ঋ","এ","ঐ","ও","ঔ","ক","ক্ক","ক্ট","ক্ত","ক্ন","ক্ব","ক্ম","ক্ল",
                                "ক্ষ","ক্ষ্ণ","ক্ষ্ব","ক্ষ্ম","ক্স","খ","গ","গ্গ","গ্দ","গ্ধ","গ্ন","গ্ব","গ্ম","গ্ল","ঘ","ঘ্ন","ঙ","ঙ্ক","ঙ্ক্ত","ঙ্ক্ষ",
                                "ঙ্খ","ঙ্গ","ঙ্ঘ","ঙ্ম","চ","চ্চ","চ্ছ","চ্ছ্ব","চ্ঞ","চ্ব","ছ","জ","জ্জ","জ্জ্ব","জ্ঝ","জ্ঞ","জ্ব","ঝ","ঞ","ঞ্চ",
                                "ঞ্ছ","ঞ্জ","ঞ্ঝ","ট","ট্ট","ট্ব","ট্ম","ঠ","ড","ড়্গ","ড্ড","ড্ব","ড্ম","ঢ","ণ","ণ্ট","ণ্ঠ","ণ্ড","ণ্ঢ","ণ্ণ",
                                "ণ্ব","ণ্ম","ত","ত্ত","ত্ত্ব","ত্থ","ত্ন","ত্ব","ত্ম","থ","থ্ব","দ","দ্গ","দ্ঘ","দ্দ","দ্দ্ব","দ্ধ","দ্ব","দ্ভ","দ্ম",
                                "ধ","ধ্ন","ধ্ব","ধ্ম","ন","ন্জ","ন্ট","ন্ঠ","ন্ড","ন্ত","ন্ত্ব","ন্থ","ন্দ","ন্দ্ব","ন্ধ","ন্ন","ন্ব","ন্ম","ন্স","ন্হ",
                                "প","প্ট","প্ত","প্ন","প্প","প্ল","প্স","ফ","ফ্ট","ফ্ফ","ফ্ল","ব","ব্জ","ব্দ","ব্ধ","ব্ব","ব্ল","ভ","ভ্ব","ভ্ল",
                                "ম","ম্ন","ম্প","ম্ফ","ম্ব","ম্ভ","ম্ম","ম্ল","য","র","ল","ল্ক","ল্গ","ল্ট","ল্ড","ল্প","ল্ফ","ল্ব","ল্ভ","ল্ম",
                                "ল্ল","শ","শ্চ","শ্ছ","শ্ন","শ্ব","শ্ম","শ্ল","ষ","ষ্ক","ষ্ট","ষ্ঠ","ষ্ণ","ষ্ণ্হ","ষ্প","ষ্ফ","ষ্ব","ষ্ম","স","স্ক",
                                "স্খ","স্ট","স্ত","স্ত্ব","স্থ","স্ন","স্প","স্ফ","স্ব","স্ম","স্ল","স্স","স্হ","হ","হ্ণ","হ্ন","হ্ব","হ্ম","হ্ল","ৎ",
                                "ড়","ড়্গ","ঢ়","য়"]
    
    # vocab
    unicodes                 =    ['']+sorted(vowels+consonants+vowel_diacritics+consonant_diacritics+special_charecters+numbers+punctuations) 
    components               =    ['']+sorted(roots+vowel_diacritics+consonant_diacritics+special_charecters+numbers+punctuations)
    valid                    =    ['']+sorted(dict_graphemes+numbers+punctuations)


#----------------------------------------- add all language info-------------------------------------------------------------
languages={}
languages["bangla"]=bangla
