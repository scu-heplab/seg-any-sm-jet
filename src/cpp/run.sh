#!/bin/bash

# Define event types and corresponding PIDs
declare -A event_pids
event_pids["pp_htt"]="25,6,-6,5,-5,24,-24"
event_pids["pp_tttt"]="6,-6,24,-24,5,-5"
# ... Add more event types here ...

# Basic Path
BASE_INPUT_PATH="path_to_your_madgraph" # "/mnt/data1/rzhang/MG5_aMC_v3_5_1"
OUTPUT_DIR="path_to_your_output_dir" # "./output"

mkdir -p "$OUTPUT_DIR"

for event_name in "${!event_pids[@]}"; do
    pids=${event_pids[$event_name]}
    input_file="${BASE_INPUT_PATH}/${event_name}/Events/run_01/tag_1_delphes_events.root"
    output_file="${OUTPUT_DIR}/${event_name}.dat"

    echo "----------------------------------------"
    echo "Processing event: $event_name"
    echo "----------------------------------------"

    if [ ! -f "$input_file" ]; then
        echo "Warning: Input file not found, skipping: $input_file"
        continue
    fi

    ./sajm.out --input "$input_file" --output "$output_file" --pids "$pids"

    echo ""
done

echo "All tasks finished."