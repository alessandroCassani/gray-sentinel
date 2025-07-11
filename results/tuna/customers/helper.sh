#!/bin/bash
echo $(pwd)

# baseline
# cpu_stress

EXPERIMENTS=("baseline" "cpu_stress" "IO pressure" "net_loss" "mem_stress")

for TARGET_EXPERIMENT in "${EXPERIMENTS[@]}"; do
    mkdir -p "unified/$TARGET_EXPERIMENT"
    BASE_DIR=$(pwd)
    
    cd "cpu related/cleaned_data"
    find "$TARGET_EXPERIMENT" -type f -exec cp -f {} "../../unified/$TARGET_EXPERIMENT" \;
    cd ../..

    cd "IO related/cleaned_disk_data"
    find "$TARGET_EXPERIMENT" -type f -exec cp -f {} "../../unified/$TARGET_EXPERIMENT" \;
    cd ../..

    cd "memory related/cleaned_memory_data"
    find "$TARGET_EXPERIMENT" -type f -exec cp -f {} "../../unified/$TARGET_EXPERIMENT" \;
    cd ../..

    cd "tcp related/cleaned_tcp_data"
    find "$TARGET_EXPERIMENT" -type f -exec cp -f {} "../../unified/$TARGET_EXPERIMENT" \;
    cd ../..

    python3 create_single.py "unified/$TARGET_EXPERIMENT" "unified/$TARGET_EXPERIMENT/all_data_${TARGET_EXPERIMENT}.csv"

    echo $(pwd)
done