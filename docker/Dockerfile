FROM nvcr.io/nvidia/pytorch:23.12-py3

# Install dependencies for tiny bert
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    vim \
    && rm -rf /var/lib/apt/lists/*

# This installation is copied from https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md

RUN pip uninstall ninja -y && pip install ninja -U


RUN MAX_JOBS=12 pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers


COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt tokenizers sentencepiece
RUN rm /workspace/requirements.txt

RUN git clone https://github.com/Dao-AILab/flash-attention
RUN cd flash-attention && \
    MAX_JOBS=16 python setup.py install && \
    cd csrc/rotary && pip install . && \
    cd ../layer_norm && pip install . && \
    cd ../xentropy && pip install . && \
    cd ../.. && rm -rf flash-attention

# RUN git clone https://github.com/timdettmers/bitsandbytes.git

# RUN cd bitsandbytes && \
#     CUDA_VERSION=123 make cuda12x -j && \
#     python setup.py install

RUN pip install bitsandbytes -U