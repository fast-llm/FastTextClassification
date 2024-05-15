
export CUDA_VISIBLE_DEVICES=6
export MODEL_NAME_OR_PATH=/platform_tech/xiongrongkang/checkpoint/ai_detect/ai_detect_v1/checkpoint-step-299
export PAD_SIZE=786
export NUM_GPUS=0
export PORT=9090
export WORKERS=10

gunicorn -c src/server/gunicorn_conf.py src.server.app:app


