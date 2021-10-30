#!/bin/sh
base_path="/home/apsisdev/ansary/DATASETS/GVU/"
bw_ref="/home/apsisdev/ansary/DATASETS/RAW/bangla_writing/converted/converted/"
bh_ref="/home/apsisdev/ansary/DATASETS/RAW/BN-HTR/"
bs_ref="/home/apsisdev/ansary/DATASETS/RAW/BanglaC/README.txt"
iit_path="/home/apsisdev/Rezwan/cvit_iiit-indic/"
num_samples=1000000
num_hw_samples=500000

#-----------------------------------------------------------------------------------------------
data_dir="${base_path}data/"
ds_path="${base_path}datasets/"
up_path="${base_path}uploads/"

bw_ds="${ds_path}bw/"
bh_ds="${ds_path}bh/"
bs_ds="${ds_path}bs/"
bn_pr_ds="${ds_path}bangla_printed/"
bn_hr_ds="${ds_path}bangla_handwritten/"


iit_bn_ref="${iit_path}bn/vocab.txt"
iit_bn_ds="${ds_path}iit.bn/"
iit_bn_rec="${iit_bn_ds}iit.bn/"

iit_gu_ref="${iit_path}gu/vocab.txt"
iit_gu_ds="${ds_path}iit.gu/"
iit_gu_rec="${iit_gu_ds}iit.gu/"

iit_ma_ref="${iit_path}ma/vocab.txt"
iit_ma_ds="${ds_path}iit.ma/"
iit_ma_rec="${iit_ma_ds}iit.ma/"

iit_od_ref="${iit_path}od/vocab.txt"
iit_od_ds="${ds_path}iit.od/"
iit_od_rec="${iit_od_ds}iit.od/"

iit_pn_ref="${iit_path}pn/vocab.txt"
iit_pn_ds="${ds_path}iit.pn/"
iit_pn_rec="${iit_pn_ds}iit.pn/"

iit_ta_ref="${iit_path}ta/vocab.txt"
iit_ta_ds="${ds_path}iit.ta/"
iit_ta_rec="${iit_ta_ds}iit.ta/"

iit_hn_ref="/home/apsisdev/Rezwan/IIIT-HW-Dev_v1/lexicon.txt"
iit_hn_ds="${ds_path}iit.hn/"
iit_hn_rec="${iit_hn_ds}iit.hn/"

#-----------------------------------bangla-----------------------------------------------
#-----------------------------------natrual---------------------------------------------
#python datasets/bangla_writing.py $bw_ref $ds_path
#python datagen.py $bw_ds "bangla"
#python datasets/boise_state.py $bs_ref $ds_path
#python datagen.py $bs_ds "bangla"
#python datasets/bn_htr.py $bh_ref $ds_path
#python datagen.py $bh_ds "bangla"
# python datasets/iit_indic.py $iit_bn_ref $ds_path
# python datagen.py $iit_bn_ds "bangla"
#-----------------------------------natrual---------------------------------------------
#-----------------------------------synthetic------------------------------------------
#python datagen_synth.py $data_dir "bangla" "printed" $ds_path --num_samples $num_samples
#python datagen_synth.py $data_dir "bangla" "handwritten" $ds_path --num_samples $num_hw_samples
#-----------------------------------synthetic------------------------------------------
#-----------------------------------bangla-----------------------------------------------


#-----------------------------------iit-indic-----------------------------------------------
# python create_vocab.py "hindi"
# python create_vocab.py "gujrati"
# python create_vocab.py "malyalam"
# python create_vocab.py "odiya"
# python create_vocab.py "panjabi"
# python create_vocab.py "tamil"


python datasets/iit_v1.py $iit_hn_ref $ds_path --iden "iit.hn"
python datagen.py $iit_hn_ds "hindi"
python datasets/iit_indic.py $iit_gu_ref $ds_path
python datagen.py $iit_gu_ds "gujrati"
python datasets/iit_indic.py $iit_ma_ref $ds_path
python datagen.py $iit_ma_ds "malyalam"
python datasets/iit_indic.py $iit_od_ref $ds_path
python datagen.py $iit_od_ds "odiya"
python datasets/iit_indic.py $iit_pn_ref $ds_path
python datagen.py $iit_pn_ds "panjabi"
python datasets/iit_indic.py $iit_ta_ref $ds_path
python datagen.py $iit_ta_ds "tamil"

python datagen_synth.py $data_dir "gujrati" "printed" $ds_path --num_samples $num_samples
python datagen_synth.py $data_dir "malyalam" "printed" $ds_path --num_samples $num_samples
python datagen_synth.py $data_dir "odiya" "printed" $ds_path --num_samples $num_samples
python datagen_synth.py $data_dir "panjabi" "printed" $ds_path --num_samples $num_samples
python datagen_synth.py $data_dir "tamil" "printed" $ds_path --num_samples $num_samples
python datagen_synth.py $data_dir "hindi" "printed" $ds_path --num_samples $num_samples


#-----------------------------------indic-----------------------------------------------

#-----------------------------------storing-----------------------------------------------
#python create_recs.py $iit_bn_ds "iit.bn" 
#python create_recs.py $bw_ds "bw" 
#python create_recs.py $bh_ds "bh" 
#python create_recs.py $bs_ds "bs" 
#python create_recs.py $bn_pr_ds "bangla_printed"
#python create_recs.py $bn_hr_ds "bangla_handwritten"
#-----------------------------------sroting-----------------------------------------------


echo succeeded