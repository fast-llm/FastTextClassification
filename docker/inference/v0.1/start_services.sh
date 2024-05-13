#!/bin/bash
mkdir -p ./logs

python ./src/server/app.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --num_gpus ${NUM_GPUS} \
    --port ${PORT} \
    --workers ${WORKERS}