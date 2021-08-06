# Bangla Synthetic Dataset
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
# Boise State
* **Boise_State_Bangla_Handwriting_Dataset_20200228.zip**  from  [**Boise State Bangla Handwriting Dataset**](https://scholarworks.boisestate.edu/saipl/1/)
* **Instructions**: 
    * unzip the file
    * corrupted zip issue:**fix zip issues with zip -FFv if needed**

* **processing**:
    * follow instruction in **datasets/boise_state.ipynb**
    * keep a track of the path printed at the end of execution

    ```python
    Example Execution
    LOG     :IMPORTANT: PATH TO USE FOR tools/process.py:/media/ansary/DriveData/Work/bengalAI/datasets/Recognition/bs
    ```

# BN-HTRd
* **Dataset.zip**  from  [**BN-HTRd: A Benchmark Dataset for Document Level Offline Bangla Handwritten Text Recognition (HTR)**](https://data.mendeley.com/datasets/743k6dm543/1)
* **Instructions**: 
    * unzip the file
    * Locked/Permission Issue: **fix permission issues with chmod/chown based on distribution**

* **processing**:
    * follow instruction in **datasets/bn_htr.ipynb**
    * keep a track of the path printed at the end of execution

    ```python
    Example Execution
    LOG     :IMPORTANT: PATH TO USE FOR tools/process.py:/media/ansary/DriveData/Work/bengalAI/datasets/Recognition/bh
    ```


# Bangla Writing
* **converted.zip** from  [**BanglaWriting: A multi-purpose offline Bangla handwriting dataset**](https://data.mendeley.com/datasets/r43wkvdk4w/1)
* **Instructions**: 
    * unzip the file
* **processing**:
    * follow instruction in **datasets/bangla_writing.ipynb**
    * keep a track of the path printed at the end of execution

    ```python
    Example Execution
    LOG     :IMPORTANT: PATH TO USE FOR tools/process.py:/media/ansary/DriveData/Work/bengalAI/datasets/Recognition/bw
    ```


# **NOTES**:
The outputs of: 

    * Bangla Writing, (bw) 
    * Boise State and (bs)
    * BN-HTRd notebooks (bh)

are considered **source data** and can be found [here](https://www.kaggle.com/nazmuddhohaansary/recognizer-source)

* These outputs needs to be processed to match the format of synthetic data.
* Each notebook ultimately creates the following structe:

```python
    ├── savepath
       ├── bX
            ├── images
            ├── data.csv

```    

# Processing (Natrual Datasets: bw,bs,bh)
* For processing the **images**/**data.csv** pairs saved in End of execution paths from the notebooks, use **tools/process.py**
* some data with garbage graphemes are dropped via manual inspection:
```
["া","্বা","্ল"], source: bw
["ভঁে"], source: bs
```
* processing any dataset also increaes the gvocab. 

**THE COMBINED PROCESSED DATASET IS AVAILABLE [HERE](https://www.kaggle.com/nazmuddhohaansary/recognizer-processed)**

# used datasets:
* **crnn tfrecords dataset** : https://www.kaggle.com/nazmuddhohaansary/pgvu-crnn-ctc-tfrecords
* **evaluation dataset**     : https://www.kaggle.com/nazmuddhohaansary/pgvu-eval-dataset (test subset of processed data)  