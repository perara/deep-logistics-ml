#!/bin/bash
#tf-nightly
#tb-nightly

Xvfb :1 -screen 0 1024x768x16 &> xvfb.log  &
ps aux | grep X

DISPLAY=:1.0
export DISPLAY

# Upgrade repo
#cd /root/deep_logistics && git pull
#cd /root/deep_logistics-ml  && git pull

# Update dependencies
#python3 -m pip install -r /root/deep_logistics/requirements.txt
#python3 -m pip install -r /root/deep_logistics-ml/requirements.txt

ls -la /

# Run
python3 /root/deep_logistics_ml/main.py --train_only > /root/logs/deep_logistics.log




