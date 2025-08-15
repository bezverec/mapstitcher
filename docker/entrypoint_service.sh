#!/bin/bash

set -e

INPUT_DIR="/data"
OUTPUT_DIR="/output"
MATCHING_ALGORITHM="${MATCHING_ALGORITHM:-loftr}"
VRAM_SIZE="${VRAM_SIZE:-8.0}"

echo "Monitoring $INPUT_DIR"

while true; do
    # exclude _incomplete or _done
    for dir in "$INPUT_DIR"/*/; do
        [ -d "$dir" ] || continue
        base="$(basename "$dir")"

        if [[ "$base" != *_incomplete && "$base" != *_done ]]; then
            echo "Found: $base"

            input_path="$INPUT_DIR/$base"
            output_file="${base}.jp2"

            echo "Processing: $input_path"
            python3 image_stitch_batch.py \
              --path "$input_path" \
              --output "$OUTPUT_DIR/$output_file" \
              --matching-algorithm "$MATCHING_ALGORITHM" \
              --vram-size "$VRAM_SIZE"

            # mark
            mv "$input_path" "${input_path}_done"
            echo "Done: ${base}_done"
        fi
    done

    sleep 5  # wait
done
