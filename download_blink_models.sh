#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOD_DIR/models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"

# 长度： 2681357077 (2.5G)
wget -c http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.bin

# 长度： 775
wget -c http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.json

# 长度：3368069200 (3.1G)
wget -c http://dl.fbaipublicfiles.com/BLINK/entity.jsonl

# 长度： 24180846943 (23G)
wget -c http://dl.fbaipublicfiles.com/BLINK/all_entities_large.t7

# 长度： 1340677176 (1.2G)
wget -c http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.bin

# 长度： 903
wget -c http://dl.fbaipublicfiles.com/BLINK/crossencoder_wiki_large.json

# 增加两个索引
#长度： 24180846637 (23G)
wget -c http://dl.fbaipublicfiles.com/BLINK//faiss_flat_index.pkl

# 长度： 30344379206 (28G)
wget -c http://dl.fbaipublicfiles.com/BLINK/faiss_hnsw_index.pkl


cd "$ROOD_DIR"
