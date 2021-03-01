# synthetic words


```python
Version: 0.0.3     
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


## Execution
* run **./main.py**
```python
    usage: Style transfer model synthetic data generation script
        [-h] [--img_height IMG_HEIGHT] [--data_dim DATA_DIM]
        data_path save_path

    positional arguments:
    data_path             Path of the data folder that contains converted and
                            raw folder from ReadMe.md)
    save_path             Path of the directory to save the dataset

    optional arguments:
    -h, --help            show this help message and exit
    --img_height IMG_HEIGHT
                            fixed height for each grapheme: default=128
    --data_dim DATA_DIM   dimension of word images: default=256

```
* Exception in execution:
```python
 #LOG     :error in creating image:183_16_0.jpg label:শ্রেষ্ঠ,
 #error:OpenCV(4.2.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:715: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'

 #LOG     :error in creating image:256_14_1.jpg label:ব,
 #error:OpenCV(4.2.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:715: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'

```
* run **./main.py**
```python
    usage: data.py [-h] data_dir save_dir

    script to create tfrecord data for style transfer

    positional arguments:
    data_dir    The path to the folder that contains test and train
    save_dir    The path to the folder where the tfrecords will be saved

    optional arguments:
    -h, --help  show this help message and exit

```
* The zipped dataset is uploaded [here](https://www.kaggle.com/nazmuddhohaansary/banglawords)
# References

* [word2grapheme](https://www.kaggle.com/reasat/extract-word-image-and-label) (@author: [Tahsin Reasat](https://www.kaggle.com/reasat))


# TODO
- [x] cleanup and merge scripts
- [x] Add **tfrecords**
- [ ] **indicwords.py**:(unstable): see [doc](/doc/indicwords.md)
    * clean empty lexicons
    * check non-existant graphemes

