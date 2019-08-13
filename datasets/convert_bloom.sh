#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to preprocess the Bloom dataset.
#
# Usage:
#   bash ./convert_bloom_data.sh
#
# The folder structure is assumed to be:

  # + bloom_folder
  #   + dataset
  #     + JPEGImages
  #     + SegmentationClass
  #     + ImageSets
  #     + Segmentation
  #   + tfrecord

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="."

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
BLOOM_ROOT="${WORK_DIR}/dataset"

SEG_FOLDER="${BLOOM_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${BLOOM_ROOT}/SegmentationClassRaw"

# Build TFRecords of the dataset.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${BLOOM_ROOT}/JPEGImages"
LIST_FOLDER="${BLOOM_ROOT}/ImageSets"

echo "Converting Bloom dataset..."
python3 ./build_bloom_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"