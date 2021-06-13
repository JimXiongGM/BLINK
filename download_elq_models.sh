#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


set -e
set -u

ROOT_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOT_DIR/models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"

# 长度： 2681381121 (2.5G)
wget -c http://dl.fbaipublicfiles.com/elq/elq_wiki_large.bin

# 长度： 2681381121 (2.5G)
wget -c http://dl.fbaipublicfiles.com/elq/elq_webqsp_large.bin

# 长度： 1601 (1.6K)
wget -c http://dl.fbaipublicfiles.com/elq/elq_large_params.txt

# 长度： 3483640967 (3.2G)
wget -c http://dl.fbaipublicfiles.com/elq/entity.jsonl

# 长度： 6045211994 (5.6G)
wget -c http://dl.fbaipublicfiles.com/elq/entity_token_ids_128.t7

# wget -c http://dl.fbaipublicfiles.com/BLINK/all_entities_large.t7

# 长度： 30320765098 (28G)
wget -c http://dl.fbaipublicfiles.com/elq/faiss_hnsw_index.pkl

cd "$ROOT_DIR"
