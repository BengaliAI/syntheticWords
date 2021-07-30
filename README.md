
# synthetic words

```python
Version: 0.1.1     
```
### **Related resources**:

For a collection of related resources in Hand written OCR research [click here](https://docs.google.com/spreadsheets/d/1LcEsd3z6lv4MO-ynbAawEjJ27jvPUoFiU9adQkD9g1A/edit?usp=sharing) 


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
* **pip requirements**: ```pip3 install -r requirements.txt``` 
> Its better to use a virtual environment 
OR use conda-
* **conda**: use environment.yml



# Datasets
## Bangla Synthetic Dataset
* For creating **synthetic bangla (numbers and letters)**,the following datasets are used:
    * The bangla **graphemes** dataset is taken from [here](https://www.kaggle.com/pestipeti/bengali-quick-eda/#data). 
    * The bangla **numbers** dataset is taken from [here](https://www.kaggle.com/c/numta/data) 
* A processed version of the dataset can be found [here](https://www.kaggle.com/nazmuddhohaansary/recognizer-source). The folder structre should look as follows
    
```python
    ├── bangla
       ├── graphemes.csv
       ├── numbers.csv
       ├── dictionary.csv
       ├── fonts
       ├── graphemes
       └── numbers
```
* To create the dataset execute **scripts/data_banglaSynth.py**

```python
    usage: Recognizer Training: Bangla Synthetic (numbers and graphemes) Dataset Creating Script [-h] [--img_height IMG_HEIGHT] [--pad_height PAD_HEIGHT] [--img_width IMG_WIDTH] [--num_samples NUM_SAMPLES]
                                                                                                data_path save_path

    positional arguments:
    data_path             Path of the source data folder
    save_path             Path of the directory to save the dataset

    optional arguments:
    -h, --help            show this help message and exit
    --img_height IMG_HEIGHT
                            height for each grapheme: default=64
    --pad_height PAD_HEIGHT
                            pad height for each grapheme for alignment correction: default=20
    --img_width IMG_WIDTH
                            width dimension of word images: default=512
    --num_samples NUM_SAMPLES
                            number of samples to create when not using dictionary:default=100000



```

**NOTES**:
* the **data_path** is the container of the unzipped bangla folder. I.E- the **source** folder should maintain the following structre:

```python
    ├── source
       ├── bangla
       ├── other random stuff
       ........................
       ........................ 
    
```
* upon execution two folders namely : **bangla.graphemes** and **bangla.numbers** will be created at the save_path.
* These folders will maintain the following structre:

```python
    ├── savepath
       ├── bangla.XXXXX
            ├── images
            ├── targets
            ├── data.csv
```
* a **vocab.json** file will be created in the working directory. This will be used to map **unicode and grapheme level** labeling along with corresponding **data.csv**
* As the datasets are added further (future), this vocab.json will change holding the "banglaSynth" vocabulary as the base case.              
* **data.csv** contains the following columns: filename,labels,image_mask,target_mask. Where labels indicate grapheme components. The **_mask** data can be used for attention based models (like [robust scanner](https://arxiv.org/abs/2007.07542))
* **targets** folder will be used for **font-face modifier model**
* any type of **cnn-lstm-ctc** model data can be created from the generated dataset

## Bangla Writing
* only download the **converted.zip** from  [**BanglaWriting: A multi-purpose offline Bangla handwriting dataset**](https://data.mendeley.com/datasets/r43wkvdk4w/1)
* unzip the file
* follow instruction in **datasets/bangla_writing.ipynb**
* keep a track of the path printed at the end of execution

```python
Example Execution
LOG     :IMPORTANT: PATH TO USE FOR tools/process.py:/media/ansary/DriveData/Work/bengalAI/datasets/Recognition/bw
```
## Boise State
* download **Boise_State_Bangla_Handwriting_Dataset_20200228.zip**  from  [**Boise State Bangla Handwriting Dataset**](https://scholarworks.boisestate.edu/saipl/1/)
* unzip the file: **fix zip issues with zip -FFv if needed**
* follow instruction in **datasets/boise_state.ipynb**
* keep a track of the path printed at the end of execution

```python
Example Execution
LOG     :IMPORTANT: PATH TO USE FOR tools/process.py:/media/ansary/DriveData/Work/bengalAI/datasets/Recognition/bs
```
## BN-HTRd
* download **Dataset.zip**  from  [**BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)**](https://data.mendeley.com/datasets/743k6dm543/1)
* unzip the file: **fix permission issues with chmod**
* follow instruction in **datasets/bn_htr.ipynb**
* keep a track of the path printed at the end of execution

```python
Example Execution
LOG     :IMPORTANT: PATH TO USE FOR tools/process.py:/media/ansary/DriveData/Work/bengalAI/datasets/Recognition/bh
```

**NOTES**:
* The outputs of Bangla Writing, Boise State and BN-HTRd notebooks are considered **source data** and can be found [here](https://www.kaggle.com/nazmuddhohaansary/recognizer-source)
* These outputs needs to be processed to match the format of synthetic data.
* Each notebook ultimately creates the following structe:

```python
    ├── savepath
       ├── bX
            ├── images
            ├── data.csv
```    


### Processing (Natrual Datasets)
* For processing the **images**/**data.csv** pairs saved in End of execution paths from the notebooks, use **tools/process.py**
* some data with garbage graphemes are dropped via manual inspection:
```
["া","্বা","্ল"], source: bw
["ভঁে"], source: bs
```
* processing any dataset also increaes the gvocab. 
* Executing **tools/process.py** (change directory in error case```cd tools```)
```python
usage: Processing Dataset Script [-h] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] data_path save_path iden

positional arguments:
  data_path             Path of the source data folder for any naturally writen images/data.csv pair dataset
  save_path             Path of the directory to save the processed dataset
  iden                  identifier of the dataset

optional arguments:
  -h, --help            show this help message and exit
  --img_height IMG_HEIGHT
                        height for each grapheme: default=64
  --img_width IMG_WIDTH
                        width dimension of word images: default=512

```

### tfrecords (from processed datasets)
* Convert Processed Dataset to tfrecords
* Executing **tools/record.py** (change directory in error case```cd tools```)
```python
usage: Script for Creating tfrecords [-h] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] data_path save_path iden

positional arguments:
  data_path             Path of the processed data folder . Should hold images,targets and data.csv
  save_path             Path of the directory to save tfrecords
  iden                  identifier of the dataset

optional arguments:
  -h, --help            show this help message and exit
  --img_height IMG_HEIGHT
                        height for each grapheme: default=64
  --img_width IMG_WIDTH
                        width dimension of word images: default=512
```
# References/Tools

* [word2grapheme](https://www.kaggle.com/reasat/extract-word-image-and-label) (@author: [Tahsin Reasat](https://www.kaggle.com/reasat))

