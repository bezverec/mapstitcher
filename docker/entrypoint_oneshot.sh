#!/bin/bash
set -e

python3 image_stitch_batch.py \
  --path /data/ \
  --output /output/"${OUTPUT_FILE}" \
  --matching-algorithm "${MATCHING_ALGORITHM}" \
  --vram-size "${VRAM_SIZE}"
