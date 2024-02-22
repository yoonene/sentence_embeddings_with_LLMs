
sudo docker run --rm -it -v /home/yoonhyek/scaling_sentemb:/workspace --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all --name cuda117_semb cuda117:kyh