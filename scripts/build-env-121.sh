# #安装环境 fasttext121
# 分步执行注释里面的创建环境
# conda deactivate
# conda create -n fasttext python=3.10 -y
# conda activate fasttext

pip install -U pip
# cudatoolkit
conda install https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/cudatoolkit-11.8.0-h4ba93d1_13.conda
# cudnn
conda install https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/cudnn-8.9.7.29-hcdd5f01_2.conda
conda install -c nvidia cuda-compiler



# https://pytorch.org/get-started/previous-versions/
# xformers版本地址: https://github.com/facebookresearch/xformers/tags
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
# https://github.com/huggingface/transformers/tags
pip install poetry
poetry init
poetry add "transformers==4.39.1" tiktoken einops transformers_stream_generator==0.0.5

poetry add torchtext scikit-learn tensorboardX nltk numpy pandas sentencepiece scipy protobuf jieba rouge-chinese nltk uvicorn pydantic fastapi sse-starlette matplotlib