# Bangla Synthetic Dataset
* The bangla **graphemes** dataset is taken from [here](https://www.kaggle.com/pestipeti/bengali-quick-eda/#data). 
* The bangla folder can be found [here](https://www.kaggle.com/nazmuddhohaansary/recognizer-source) with some additional fonts
* filtered font list 

```python

    ├── AdorshoLipi_20-07-2007.ttf
    ├── akaashnormal.ttf
    ├── Bangla.ttf
    ├── BenSenHandwriting.ttf
    ├── BenSen.ttf
    ├── kalpurush.ttf
    ├── mitra.ttf
    ├── Mukti_1.99_PR.ttf
    ├── muktinarrow.ttf
    ├── NikoshBAN.ttf
    ├── NikoshGrameen.ttf
    ├── NikoshLightBan.ttf
    ├── NikoshLight.ttf
    ├── Nikosh.ttf
    ├── sagarnormal.ttf
    ├── Siyamrupali.ttf
    └── SolaimanLipi_20-04-07.ttf

```



The  **required** folder structre should look as follows
    
```python
    ├── bangla
       ├── fonts
       ├── graphemes.csv
       ├── graphemes
       ├── numbers.csv
       ├── numbers
       
```


# Boise State
* **Boise_State_Bangla_Handwriting_Dataset_20200228.zip**  from  [**Boise State Bangla Handwriting Dataset**](https://scholarworks.boisestate.edu/saipl/1/)
* **Instructions**: 
    * unzip the file
    * corrupted zip issue:**fix zip issues with zip -FFv if needed**
* Not Found

```
['\u200c', 'প্ব', 'ক্ষ্ন', 'স্প্ল', 'ল্\u200c', 'ম্ষ', 'ছ্ব', 'ষ্ন', 'ল্থ']

```

# BN-HTRd
* **Dataset.zip**  from  [**BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)**](https://data.mendeley.com/datasets/743k6dm543/1)
* **Instructions**: 
    * unzip the file
    * Locked/Permission Issue: **fix permission issues with chmod/chown based on distribution**
    * **ClearDirectFormatting for 34.xlsx**
* Not found

```
[' ', '’', '2', '0', '‘', '1', '4', 'ৗ', '়', '6', '5', '3', '8', '9', '7', 'B', 'C', 'N', 'e', 'w', 's', 'G', 't', 'y', 'I', 'm', 'a', 'g', '\n', '©', '•', 'O', 'V', 'D', '—', '“', 'o', 'n', 'r', 'i', 'b', 'u', 'A', 'F', 'P', 'h', 'S', 'H', 'l', 'v', 'R', 'K', 'c', 'E', 'M', 'd', 'Y', 'L', 'p', 'k', 'T', '\u200c', 'W', 'U', 'Z', 'J', 'z', '৷', 'f', 'Q', '”', '–', '·', 'X', 'x', 'আ্ও', 'ক্ন্ব', 'ু্\u200c', 'ু্ব', 'শ্ট', 'ক্ ', 'হ্ ', '়্গ', 'স্ ', 'ন্ ', 'জ্ন', 'ে্গ', 'ল্থ']
```

# Bangla Writing
* **converted.zip** from  [**BanglaWriting: A multi-purpose offline Bangla handwriting dataset**](https://data.mendeley.com/datasets/r43wkvdk4w/1)
* **Instructions**: 
    * unzip the file
* Not Found:

```
    ['f', 'u', 'n', 'd', 'a', 'm', 'e', 't', 'l', 'i', 'r', 'c', 'o', 
    'S', 'T', '3', ' ', '1', '5', '9', '6', '0', 'X', 'x', '2', 'v', 
    'O', 'θ', 'π', 'I', 'A', 'g', 'J', 'h', 'গ্ণ', 'া্ব', 'ূ্ল']
```

# iit-indic
* Download the dataset from [here](http://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words)
* unzip all the directories and keep track of vocab.txt

## **bn**
* not found

```
['ৗ', '়', '৵', '৹', 'ঽ', 'ৱ', '৴', 'ৄ', '৶', 'ৰ', '৷', 'ণ্ন', 'ি্ন', 'ত্ক', 'হ্উ']
```

# **NOTES**:
Output Structre

```python
    ├── savepath
       ├── bX
            ├── images
            ├── data.csv

```    

