#!/bin/sh
python /home/apsisdev/ansary/CODES/syntheticWords/scripts/data_englishSynth.py  /home/apsisdev/ansary/DATASETS/Recognition/source/ /home/apsisdev/ansary/DATASETS/Recognition/synth_font && \
python /home/apsisdev/ansary/CODES/syntheticWords/scripts/data_fontfaced.py  /home/apsisdev/ansary/DATASETS/Recognition/source/ /home/apsisdev/ansary/DATASETS/Recognition/synth_font 
echo succeeded
