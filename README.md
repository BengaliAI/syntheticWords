
# synthetic words

```python
Version: 0.0.8     
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
* ```pip3 install -r requirements.txt``` 
> Its better to use a virtual environment 




# Synthetic words
## Dataset
* The **source data** folder can be found [here](https://www.kaggle.com/nazmuddhohaansary/recogbnsrcbsbwbh)



## RecognizerTraining
 
* run **./main.py**
```python

    usage: Recognizer Training Dataset Creating Script [-h]
                                                    [--img_height IMG_HEIGHT]
                                                    [--img_width IMG_WIDTH]
                                                    [--num_samples NUM_SAMPLES]
                                                    data_path save_path

    positional arguments:
    data_path             Path of the source data folder 
    save_path             Path of the directory to save the dataset

    optional arguments:
    -h, --help            show this help message and exit
    --img_height IMG_HEIGHT
                            height for each grapheme: default=32
    --img_width IMG_WIDTH
                            width dimension of word images: default=256
    --num_samples NUM_SAMPLES
                            number of samples to create per word: default=250


```

         
* Exception in execution:

```python
 #LOG     :error in creating image:183_16_0.jpg label:শ্রেষ্ঠ,
 #error:OpenCV(4.2.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:715: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'

 #LOG     :error in creating image:256_14_1.jpg label:ব,
 #error:OpenCV(4.2.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:715: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'

```

# References

* [word2grapheme](https://www.kaggle.com/reasat/extract-word-image-and-label) (@author: [Tahsin Reasat](https://www.kaggle.com/reasat))

