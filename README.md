# related resources

For a collection of related resources in Hand written OCR research [click here](https://docs.google.com/spreadsheets/d/1LcEsd3z6lv4MO-ynbAawEjJ27jvPUoFiU9adQkD9g1A/edit?usp=sharing) 

# synthetic words


```python
Version: 0.0.4     
Authors: Md. Nazmuddoha Ansary,
        Tahsin Reasat,
        Imtiaz Prio,
        Sushmit Asif  
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
>Assuming the **libraqm** complex layout is working properly, you can skip to **python requirements**. 
*  ```sudo apt-get install libfreetype6-dev libharfbuzz-dev libfribidi-dev gtk-doc-tools```
* Install libraqm as described [here](https://github.com/HOST-Oman/libraqm)
* ```sudo ldconfig``` (librarqm local repo)

**python requirements**

* ```pip3 install pillow --global-option="build_ext" --global-option="--enable-freetype"```
* ```pip3 install -r requirements.txt``` 
> Its better to use a virtual environment 



# Synthetic words
* download the **words.csv** from [here](https://www.kaggle.com/reasat/extract-word-image-and-label) and keep in under **resources** folder

**caution**: Do not change anything under **resources**. The directory should look as follows
```python
    ├── classes.csv
    ├── font.ttf
    ├── label.csv
    └── words.csv
```


## Dataset
* The **writting** dataset is taken from [here](https://www.kaggle.com/reasat/banglawriting).
    * only the **converted**  sub-folder within the **converted** folder is used as the input data 

* The **grapheme** dataset is taken from [here](https://www.kaggle.com/pestipeti/bengali-quick-eda/#data). 
    * Only the **256** folder under **256_train** is kept and renamed as **RAW** form **BengaliAI:Supplementary dataset for BengaliAI Competition**

* The final **data** folder is like as follows:

```python
    data
    ├── converted
    └── RAW
```
## RecognizerTraining 
* run **./main.py**
```python

    usage: Recognizer Training Dataset Creating Script [-h]
                                                    [--img_height IMG_HEIGHT]
                                                    [--img_width IMG_WIDTH]
                                                    [--num_samples NUM_SAMPLES]
                                                    data_path save_path

    positional arguments:
    data_path             Path of the data folder that contains converted and
                            raw folder from ReadMe.md)
    save_path             Path of the directory to save the dataset

    optional arguments:
    -h, --help            show this help message and exit
    --img_height IMG_HEIGHT
                            height for each grapheme: default=32
    --img_width IMG_WIDTH
                            width dimension of word images: default=128
    --num_samples NUM_SAMPLES
                            number of samples to create per word: default=10


```
* Exception in execution:
```python
 #LOG     :error in creating image:183_16_0.jpg label:শ্রেষ্ঠ,
 #error:OpenCV(4.2.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:715: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'

 #LOG     :error in creating image:256_14_1.jpg label:ব,
 #error:OpenCV(4.2.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:715: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'

```

**Execution Result**:
* creates **train.csv** and **test.csv**  which holds image id,label and graphemes
* creates a folder callder **data** in **save_path** where :
    * **train** folder contains the training images
    * **test** folder contains the testing images



# References

* [word2grapheme](https://www.kaggle.com/reasat/extract-word-image-and-label) (@author: [Tahsin Reasat](https://www.kaggle.com/reasat))
