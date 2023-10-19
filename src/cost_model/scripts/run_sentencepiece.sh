#!/bin/bash

set -xe

spm_train --input=$1 --model_prefix=cbench_arm64_assembly_with_line_no --vocab_size=1000 --character_coverage=1.0 --model_type=bpe