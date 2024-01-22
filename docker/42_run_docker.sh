#!/bin/bash

docker run -v /home/xiachunwei/Projects/TinyLlama:/workspace/TinyLlama \
    -v /data0/xiachunwei/Dataset:/workspace/Dataset \
    -it --rm --runtime=nvidia --gpus all \
    llm4compiler:23.12 /bin/bash
