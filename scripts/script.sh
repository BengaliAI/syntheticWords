#!/bin/sh
base_path="/home/apsisdev/ansary/DATASETS/GVU/"
bw_ref="/home/apsisdev/ansary/DATASETS/RAW/bangla_writing/converted/converted/"
bh_ref="/home/apsisdev/ansary/DATASETS/RAW/BN-HTR/"
bs_ref="/home/apsisdev/ansary/DATASETS/RAW/BanglaC/README.txt"
iit_path="/home/apsisdev/Rezwan/cvit_iiit-indic/"

#iit_path="/media/ansary/DriveData/Work/APSIS/datasets/__raw__/bengal/iiit-indic/"
#base_path="/media/ansary/DriveData/Work/APSIS/datasets/GVU/"
#-----------------------------------------------------------------------------------------------
data_dir="${base_path}data/"
ds_path="${base_path}datasets/"
bw_ds="${ds_path}bw/"
bh_ds="${ds_path}bh/"
bs_ds="${ds_path}bs/"
bn_pr_ds="${ds_path}bangla_printed/"
bn_hr_ds="${ds_path}bangla_handwritten/"


iit_bn_ref="${iit_path}bn/vocab.txt"
iit_bn_ds="${ds_path}iit.bn/"
iit_bn_test_ds="${ds_path}iit.bn/test/"

iit_gu_ref="${iit_path}gu/vocab.txt"
iit_gu_ds="${ds_path}iit.gu/"

iit_kn_ref="${iit_path}kn/vocab.txt"
iit_kn_ds="${ds_path}iit.kn/"

iit_ma_ref="${iit_path}ma/vocab.txt"
iit_ma_ds="${ds_path}iit.ma/"

iit_od_ref="${iit_path}od/vocab.txt"
iit_od_ds="${ds_path}iit.od/"

iit_pn_ref="${iit_path}pn/vocab.txt"
iit_pn_ds="${ds_path}iit.pn/"

iit_ta_ref="${iit_path}ta/vocab.txt"
iit_ta_ds="${ds_path}iit.ta/"

iit_ur_ref="${iit_path}ur/vocab.txt"
iit_ur_ds="${ds_path}iit.ur/"


#-----------------------------------bangla-----------------------------------------------
#-----------------------------------natrual---------------------------------------------
#python datasets/bangla_writing.py $bw_ref $ds_path
#python datagen.py $bw_ds "bangla"
#python datasets/boise_state.py $bs_ref $ds_path
#python datagen.py $bs_ds "bangla"
#python datasets/bn_htr.py $bh_ref $ds_path
#python datagen.py $bh_ds "bangla"
python datasets/iit_indic.py $iit_bn_ref $ds_path
python datagen.py $iit_bn_ds "bangla"
python datagen.py $iit_bn_test_ds "bangla" --iden "iit.bn.test"
#-----------------------------------natrual---------------------------------------------
#-----------------------------------synthetic------------------------------------------
#python datagen_synth.py $data_dir "bangla" "printed" $ds_path --num_samples 1000000
#python datagen_synth.py $data_dir "bangla" "handwritten" $ds_path --num_samples 500000
#-----------------------------------synthetic------------------------------------------
#-----------------------------------bangla-----------------------------------------------

#-----------------------------------indic-----------------------------------------------
python datasets/iit_indic.py $iit_gu_ref $ds_path
python datasets/iit_indic.py $iit_kn_ref $ds_path
python datasets/iit_indic.py $iit_ma_ref $ds_path
python datasets/iit_indic.py $iit_od_ref $ds_path
python datasets/iit_indic.py $iit_pn_ref $ds_path
python datasets/iit_indic.py $iit_ta_ref $ds_path
python datasets/iit_indic.py $iit_ur_ref $ds_path
#-----------------------------------indic-----------------------------------------------

#-----------------------------------storing-----------------------------------------------
python create_recs.py $iit_bn_ds "iit.bn" 
#python create_recs.py $bw_ds "bw" 
#python create_recs.py $bh_ds "bh" 
#python create_recs.py $bs_ds "bs" 
#python create_recs.py $bn_pr_ds "bangla_printed"
#python create_recs.py $bn_hr_ds "bangla_handwritten"
#-----------------------------------sroting-----------------------------------------------


echo succeeded