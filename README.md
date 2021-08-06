
# synthetic words

```python
Version: 0.1.2     
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
# Environment Setup
>Assuming the **libraqm** complex layout is working properly, you can skip to **python requirements**. 
*  ```sudo apt-get install libfreetype6-dev libharfbuzz-dev libfribidi-dev gtk-doc-tools```
* Install libraqm as described [here](https://github.com/HOST-Oman/libraqm)
* ```sudo ldconfig``` (librarqm local repo)

**python requirements**
* **pip requirements**: ```pip install -r requirements.txt``` 
> Its better to use a virtual environment 
OR use conda-
* **conda**: use environment.yml: ```conda env create -f environment.yml```


# Resources:
* **Bangla.ttf**        : Font to create synthetic font-faced data
* **vocab.json**        : holder for unicode and grapheme level vocabulary data
* **scripts(folder)**   : fixed-execution (the execution is for certain purposes) **.py** files
* **tools(folder)**     : **.py** files created with generalized / variable execution purposes
* **datasets(folder)**  : **.ipynb** files with specific instructions for specific natrual datasets
* **datasets.md**       : sources and processing instuctions for used datasets  
* **execution.md**      : execution notes for all scripts and tools
* **coreLib(folder)**   : a custom module for cleaner work flow


# Re-production

The series of execution is as follows

1. **scripts/data_banglaSynth.py**
2. **datasets/boise_state.ipynb**
3. **datasets/bn_htr.ipynb**
4. **datasets/bangla_writing.ipynb**
5. **tools/process.py** x 3 : execute for all 3 natrual written datasets(bw,bs,bh)
6. **scripts/extend_vocab.py**
7. **tools/record.py**  x 8 : to create tfrecords for the following datasets

```
    bangla.graphemes
    bangla.numbers
    bw/train
    bw/test
    bs/train
    bs/test
    bh/train
    bh/test
    
```
**se execution.md for notes and specific instructions** 





