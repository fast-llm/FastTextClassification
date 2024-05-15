export version=v0.1
export docker_name=classification-worker
sudo docker stop ${docker_name} && sudo docker rm ${docker_name}
export CUDA_VISIBLE_DEVICES=2
export PORT=9090
export MODEL_NAME_OR_PATH=/platform_tech/xiongrongkang/checkpoint/ai_detect/ai_detect_v3/best_model

sudo docker run --rm --security-opt seccomp:unconfined -e OPENBLAS_NUM_THREADS=1 \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    -e TZ=Asia/Shanghai \
    -e PYTHONPATH="/app/src" \
    -e NUM_GPUS=1 \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -p ${PORT}:9090 \
    -v /etc/localtime:/etc/localtime:ro \
    -v $MODEL_NAME_OR_PATH:/app/model \
    -v /home/xiongrongkang/WorkSpace/Code/AcademicGPT/FastTextClassification/logs:/app/logs \
    --name ${docker_name} classification-worker:${version}
        # -v /home/xiongrongkang/WorkSpace/Code/AcademicGPT/FastTextClassification/src:/app/src \
