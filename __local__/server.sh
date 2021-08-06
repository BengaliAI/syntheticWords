#!/bin/sh
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bw/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bw.test CRNN && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bs/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bs.test CRNN && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bh/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bh.test CRNN && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bw/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bw.train CRNN && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bs/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bs.train CRNN && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bh/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bh.train CRNN && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bangla.graphemes/ /home/apsisdev/ansary/DATASETS/Recognition/ synth.bn.words CRNN && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bangla.numbers/ /home/apsisdev/ansary/DATASETS/Recognition/ synth.bn.numbers CRNN
echo succeeded
