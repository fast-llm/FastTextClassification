FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
ARG BUILD_DATE=unknown
RUN echo "Build date: ${BUILD_DATE}"
RUN python3 --version
# 设置pip源为清华大学镜像源
WORKDIR /app
COPY . /app

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip config --user set global.progress_bar off && \
    pip install --upgrade pip && \
    pip install "transformers==4.39.1" tiktoken einops transformers_stream_generator==0.0.5 && \
    pip install 'torchtext==0.17.0' scikit-learn nltk numpy pandas sentencepiece scipy protobuf jieba rouge-chinese uvicorn pydantic fastapi sse-starlette && \
    pip install GPUtil numba


ENV MODEL_NAME_OR_PATH=/app/model \
    NUM_GPUS=1 \
    PORT=9090 \
    WORKERS=1

RUN chmod +x ./docker/inference/v0.1/start_services.sh

ENTRYPOINT ["./docker/inference/v0.1/start_services.sh"]