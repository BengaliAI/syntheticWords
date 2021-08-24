#!/bin/sh
python /home/apsisdev/ansary/CODES/syntheticWords/tools/process.py  /home/apsisdev/ansary/DATASETS/PRecog/source/bw/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bw --ptype central && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/process.py  /home/apsisdev/ansary/DATASETS/PRecog/source/bs/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bs --ptype central && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/process.py  /home/apsisdev/ansary/DATASETS/PRecog/source/bh/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bh --ptype central && \
python /home/apsisdev/ansary/CODES/syntheticWords/scripts/extend_vocab.py && \
python /home/apsisdev/ansary/CODES/syntheticWords/scripts/data_synth.py /home/apsisdev/ansary/DATASETS/PRecog/source/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ --pad_type central && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/crnn/bw/test/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bw.test CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/crnn/bs/test/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bs.test CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/crnn/bh/test/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bh.test CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/crnn/bw/train/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bw.train CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/crnn/bs/train/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bs.train CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/crnn/bh/train/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ bh.train CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/PRecog/crnn/synth/ /home/apsisdev/ansary/DATASETS/PRecog/crnn/ synth CRNN --max_glen 10 --max_clen 20
echo succeeded