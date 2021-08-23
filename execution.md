#  **tools/process.py** 

* For processing the **images** and **data.csv** saved in End of execution paths from the datasets notebooks
* change directory: ```cd tools```
* execution params:

```python
usage: Processing Dataset Script [-h] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] [--ptype PTYPE] data_path save_path iden

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
  --ptype PTYPE         type of padding to use(for CRNN use central , for ROBUSTSCANNER use left): default=left

```

# **tools/record.py**

* For create records from the **images** and **data.csv** saved in End of execution paths from the **tools/process.py**
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