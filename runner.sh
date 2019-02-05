#!/bin/bash
#tf-nightly
#tb-nightly

Xvfb :1 -screen 0 1024x768x16 &> xvfb.log  &
ps aux | grep X

DISPLAY=:1.0
export DISPLAY

# Upgrade repo
#cd /root/deep-logistics && git pull
#cd /root/deep-logistics-ml  && git pull

# Update dependencies
#python3 -m pip install -r /root/deep-logistics/requirements.txt
#python3 -m pip install -r /root/deep-logistics-ml/requirements.txt

ls -la /

# Run
python3 /root/deep-logistics-ml/main.py --train_only > /root/logs/deep-logistics.log




