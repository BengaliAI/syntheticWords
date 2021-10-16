#!/bin/sh
base_path="/media/ansary/DriveData/Work/APSIS/datasets/Recognition/"
#-----------------------------------------------------------------------------------------------
data_dir="${base_path}data/"
ds_path="${base_path}datasets/"
#-----------------------------------bangla-----------------------------------------------
python datagen_synth.py $data_dir "bangla" "printed" $ds_path --num_samples 10
python datagen_synth.py $data_dir "bangla" "handwritten" $ds_path --num_samples 10
#-----------------------------------bangla-----------------------------------------------
echo succeeded