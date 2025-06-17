#!/bin/bash

# Define event types and corresponding PIDs
declare -A event_pids
event_pids["htt"]="25,6,-6,5,-5,24,-24"
event_pids["tttt"]="6,-6,24,-24,5,-5"
# ... Add more event types here ...

# Basic Path
INPUT_DIR="path_to_your_root_dir" # "./input"
OUTPUT_DIR="path_to_your_output_dir" # "./output"

mkdir -p "$OUTPUT_DIR"

for event_name in "${!event_pids[@]}"; do
    pids=${event_pids[$event_name]}
    input_file="${INPUT_DIR}/${event_name}.root"
    output_file="${OUTPUT_DIR}/${event_name}.dat"

    echo "----------------------------------------"
    echo "Processing event: $event_name"
    echo "----------------------------------------"

    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file not found, skipping: $input_file"
        continue
    fi

    ./sajm --input "$input_file" --output "$output_file" --pids "$pids"

    echo ""
done

echo "All tasks finished."