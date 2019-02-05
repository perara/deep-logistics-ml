#!/bin/bash
#tf-nightly
#tb-nightly

Xvfb :1 -screen 0 1024x768x16 &> xvfb.log  &
ps aux | grep X

DISPLAY=:1.0
export DISPLAY

# Upgrade repo
cd deep-logistics && git pull && cd ..
cd deep-logistics-ml  && git pull && cd ..

# Update dependencies
python3 -m pip install -r $PWD/deep-logistics/requirements.txt
python3 -m pip install -r $PWD/deep-logistics-ml/requirements.txt

# Run
python3 $PWD/deep-logistics-ml/main.py
