#!/bin/sh
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bw/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bw.test CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bs/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bs.test CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bh/test/ /home/apsisdev/ansary/DATASETS/Recognition/ bh.test CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bw/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bw.train CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bs/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bs.train CRNN --max_glen 10 --max_clen 20 && \
python /home/apsisdev/ansary/CODES/syntheticWords/tools/record.py  /home/apsisdev/ansary/DATASETS/Recognition/processed/bh/train/ /home/apsisdev/ansary/DATASETS/Recognition/ bh.train CRNN --max_glen 10 --max_clen 20
echo succeeded