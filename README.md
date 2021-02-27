# synthetic words


```python
Version: 0.0.2     
Authors: Md. Nazmuddoha Ansary,Tahsin Reasat,Imtiaz Prio,Sushmit Asif  
```
**LOCAL ENVIRONMENT**  
```python
OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
Memory      : 7.7 GiB  
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.28.2  
```
# Setup
* ```pip3 install -r requirements.txt```

# Entry points

## indicwords
### Dataset
* The dataset is taken from [here](https://www.kaggle.com/pestipeti/bengali-quick-eda/#data). 
* Only the **256** folder under **256_train** is kept and renamed as **RAW** form **BengaliAI:Supplementary dataset for BengaliAI Competition**
* And the **train.csv** under **Bengali.AI Handwritten Grapheme Classification:Classify the components of handwritten Bengali** is renamed as **label.csv**
* **indicword_bengali_lexicon_grapheme.csv** is collected from [here](https://www.kaggle.com/reasat/indicword?select=indicword_bengali_lexicon_grapheme.csv)
* The final **data** folder is like as follows:
```python
    data
    ├── label.csv
    ├── indicword_bengali_lexicon_grapheme.csv
    └── RAW
```

* run **./indicwords**
```python
    usage: synthetic word generation script [-h] [--sample_num SAMPLE_NUM]
                                            [--img_height IMG_HEIGHT]
                                            [--data_dim DATA_DIM]
                                            data_path save_iden

    positional arguments:
    data_path             Path of the data folder that contains label.csv,RAW
                            folder and indicword_lexicon_grapheme.csv
    save_iden             identifier of the folfer to save data

    optional arguments:
    -h, --help            show this help message and exit
    --sample_num SAMPLE_NUM
                            number of samples to create : default=1000
    --img_height IMG_HEIGHT
                            fixed height for each grapheme: default=128
    --data_dim DATA_DIM   dimension of word images: default=256

```
**SAMPLES**
* Lexicon: **করতে** 

![](/src_imgs/0.png?raw=true)
![](/src_imgs/1.png?raw=true)
![](/src_imgs/2.png?raw=true)
![](/src_imgs/3.png?raw=true)
![](/src_imgs/4.png?raw=true)

* Lexicon: **ইনস্টিটিউটে** 

![](/src_imgs/5.png?raw=true)
![](/src_imgs/6.png?raw=true)
![](/src_imgs/7.png?raw=true)
![](/src_imgs/8.png?raw=true)
![](/src_imgs/9.png?raw=true)

## en1_images2words
### Dataset
* The dataset is taken from [here](https://www.kaggle.com/reasat/banglawriting).
* only the **converted**  sub-folder within the **converted** folder is used as the input data 
** Notebooks Used as Reference**
* https://www.kaggle.com/reasat/extract-word-image-and-label
* run **./en1_images2words**
```python
    usage: image to word generation script for banglawritting dataset
        [-h] data_path save_path

    positional arguments:
    data_path   Path of the data folder that contains .jpg s and .json s
    save_path   Path of the directory to save the images per their labels

    optional arguments:
    -h, --help  show this help message and exit
```
* Exception: **"াআমি"** label (This is corrected via copying the single image manually) 
* cleaned up data contains 5084 unique words
* Exception in execution:
```python
 #LOG     :error in creating image:183_16_0.jpg label:শ্রেষ্ঠ,
 #error:OpenCV(4.2.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:715: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'

 #LOG     :error in creating image:256_14_1.jpg label:ব,
 #error:OpenCV(4.2.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:715: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'

```






# TODO
- [ ] cleanup and merge scripts
- [ ] Add **h5** and **tfrecords**
- [ ] **indicwords.py**:
    * clean empty lexicons
    * check non-existant graphemes

