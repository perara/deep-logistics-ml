#!/usr/bin/env bash
chmod +x runner.sh

docker ps -a | awk '{ print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13}' | grep deep-logistics | awk '{print $1 }' | xargs -I {} docker rm -f {}

#nvidia-docker run -d --name deep-logistics --volume deep-logistics perara/deep-logistics:latest
nvidia-docker run \
-d \
--name deep-logistics \
--volume deep-logistics \
-v /raid/home/perara12/git/deep-logistics-ml:/root/deep-logistics-ml \
-v /raid/home/perara12/git/deep-logistics:/root/deep-logistics \
-e NVIDIA_VISIBLE_DEVICES=0,1 \
perara/deep-logistics:latest


sleep 2
docker logs --follow deep-logistics
