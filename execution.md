#  **scripts/data_banglaSynth.py**
* for creating synthetic data
* change directory: ```cd scripts```
* execution params:

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
* As the datasets are added further, this vocab.json will change holding the "gvocab" vocabulary as the base case.              
* **data.csv** contains the following columns: filename,labels,image_mask,target_mask. Where labels indicate grapheme components. 
* The **_mask** data can be used for attention based models (like [robust scanner](https://arxiv.org/abs/2007.07542))
* **targets** folder will be used for **font-face modifier model**
* any type of **cnn-lstm-ctc** model data can be created from the generated dataset



#  **tools/process.py** 

* For processing the **images** and **data.csv** saved in End of execution paths from the datasets notebooks
* change directory: ```cd tools```
* execution params:

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

# **tools/record.py**

* For create records from the **images**,**targets** and **data.csv** saved in End of execution paths from the **tools/process.py**
* change directory: ```cd tools```
* execution params:

```python
usage: Script for Creating tfrecords [-h] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] [--max_glen MAX_GLEN] [--max_clen MAX_CLEN] [--tf_size TF_SIZE] [--factor FACTOR] [--use_font USE_FONT]
                                     data_path save_path iden record_type

positional arguments:
  data_path             Path of the processed data folder . Should hold images,targets and data.csv
  save_path             Path of the directory to save tfrecords
  iden                  identifier of the dataset
  record_type           specific record type to create. Availabe['CRNN','ROBUSTSCANNER','ABINET']

optional arguments:
  -h, --help            show this help message and exit
  --img_height IMG_HEIGHT
                        height for each grapheme: default=64
  --img_width IMG_WIDTH
                        width dimension of word images: default=512
  --max_glen MAX_GLEN   maximum length of grapheme level data to keep: default=36
  --max_clen MAX_CLEN   maximum length of unicode level data to keep: default=62
  --tf_size TF_SIZE     number of data to keep in one record: default=1024
  --factor FACTOR       downscale factor for attention mask(used in robust scanner and abinet): default=32
  --use_font USE_FONT   Stores fontface images: default=False

```