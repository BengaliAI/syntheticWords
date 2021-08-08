#!/bin/sh
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bw/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bw.test ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bs/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bs.test ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bh/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bh.test ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bw/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bw.train ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bs/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bs.train ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bh/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bh.train ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bangla.graphemes/ /home/apsisdev/ansary/DATASETS/Recognition/ synth.bn.words ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bangla.numbers/ /home/apsisdev/ansary/DATASETS/Recognition/ synth.bn.numbers ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/english.graphemes/ /home/apsisdev/ansary/DATASETS/Recognition/ synth.en.words ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/english.numbers/ /home/apsisdev/ansary/DATASETS/Recognition/ synth.en.numbers ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bangla.fontfaced/ /home/apsisdev/ansary/DATASETS/Recognition/ ff.bn ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/english.fontfaced/ /home/apsisdev/ansary/DATASETS/Recognition/ ff.en ROBUSTSCANNER
echo succeeded
