export version=v0.2
export docker_name=classification-worker
sudo docker stop ${docker_name} && sudo docker rm ${docker_name}
export CUDA_VISIBLE_DEVICES=6
export MODEL_NAME_OR_PATH=/platform_tech/xiongrongkang/checkpoint/ai_detect/ai_detect_v1/checkpoint-step-299
export PAD_SIZE=786
export NUM_GPUS=0
export PORT=9090
export WORKERS=2

sudo docker run --rm --security-opt seccomp:unconfined \
    --gpus all \
    -e NUM_GPUS=1 \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -p ${PORT}:9090 \
    -v /etc/localtime:/etc/localtime:ro \
    -v $MODEL_NAME_OR_PATH:/app/model \
    --name ${docker_name} classification-worker:${version}
    # -v /home/xiongrongkang/WorkSpace/Code/AcademicGPT/FastTextClassification/src:/app/src \
