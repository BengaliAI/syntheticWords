#!/bin/sh
python /home/apsisdev/ansary/CODES/syntheticWords/tools/process.py  /home/apsisdev/ansary/DATASETS/PRecog/source/bw/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bw  && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/process.py  /home/apsisdev/ansary/DATASETS/PRecog/source/bs/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bs  && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/process.py  /home/apsisdev/ansary/DATASETS/PRecog/source/bh/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bh  && \
python /home/apsisdev/ansary/CODES/syntheticWords/scripts/extend_vocab.py
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/rs/bw/test/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bw.test  ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/rs/bs/test/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bs.test  ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/rs/bh/test/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bh.test  ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/rs/bw/train/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bw.train  ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/rs/bs/train/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bs.train  ROBUSTSCANNER && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/rs/bh/train/ /home/apsisdev/ansary/DATASETS/PRecog/rs/ bh.train  ROBUSTSCANNER
echo succeeded