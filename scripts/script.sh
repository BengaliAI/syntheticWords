#!/bin/sh
#base_path="/media/ansary/DriveData/Work/APSIS/datasets/GVU/"
base_path="/home/apsisdev/ansary/DATASETS/GVU/"
bw_ref="/home/apsisdev/ansary/DATASETS/RAW/bangla_writing/converted/converted/"
bh_ref="/home/apsisdev/ansary/DATASETS/RAW/BN-HTR/"
bs_ref="/home/apsisdev/ansary/DATASETS/RAW/BanglaC/README.txt"
#-----------------------------------------------------------------------------------------------
data_dir="${base_path}data/"
ds_path="${base_path}datasets/"
bw_ds="${ds_path}bw/"
bh_ds="${ds_path}bh/"
bs_ds="${ds_path}bs/"
#-----------------------------------bangla-----------------------------------------------
#-----------------------------------natrual---------------------------------------------
python datasets/bangla_writing.py $bw_ref $ds_path
python datagen.py $bw_ds "bangla"
python datasets/boise_state.py $bs_ref $ds_path
python datagen.py $bs_ds "bangla"
python datasets/bn_htr.py $bh_ref $ds_path
python datagen.py $bh_ds "bangla"
#-----------------------------------natrual---------------------------------------------
#-----------------------------------synthetic------------------------------------------
python datagen_synth.py $data_dir "bangla" "printed" $ds_path --num_samples 1000000
python datagen_synth.py $data_dir "bangla" "handwritten" $ds_path --num_samples 500000
#-----------------------------------synthetic------------------------------------------
#-----------------------------------bangla-----------------------------------------------
echo succeeded