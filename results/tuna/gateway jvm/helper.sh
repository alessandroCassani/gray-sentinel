#!/bin/bash
echo $(pwd)

# baseline
# cpu_stress

EXPERIMENTS=("cache_filling" "retrans_packets" "oom" "threadfull")

for TARGET_EXPERIMENT in "${EXPERIMENTS[@]}"; do
    mkdir -p unified/$TARGET_EXPERIMENT
    BASE_DIR=$(pwd)
    
    # Copy from cpu related/cleaned_data
    cd "cpu related/cleaned_data"
    if [ -d "$TARGET_EXPERIMENT" ]; then
        find "$TARGET_EXPERIMENT" -type f -name "*.csv" -exec cp -f {} ../../unified/$TARGET_EXPERIMENT/ \;
    fi
    cd ../..

    # Copy from IO related/cleaned_data
    cd "IO related/cleaned_data"
    if [ -d "$TARGET_EXPERIMENT" ]; then
        find "$TARGET_EXPERIMENT" -type f -name "*.csv" -exec cp -f {} ../../unified/$TARGET_EXPERIMENT/ \;
    fi
    cd ../..

    # Copy from memory related/cleaned_data
    cd "memory related/cleaned_data"
    if [ -d "$TARGET_EXPERIMENT" ]; then
        find "$TARGET_EXPERIMENT" -type f -name "*.csv" -exec cp -f {} ../../unified/$TARGET_EXPERIMENT/ \;
    fi
    cd ../..

    # Copy from tcp related/cleaned_data
    cd "tcp related/cleaned_data"
    if [ -d "$TARGET_EXPERIMENT" ]; then
        find "$TARGET_EXPERIMENT" -type f -name "*.csv" -exec cp -f {} ../../unified/$TARGET_EXPERIMENT/ \;
    fi
    cd ../..

    # Only run Python script if the unified directory has CSV files
    if [ "$(find "unified/$TARGET_EXPERIMENT" -name "*.csv" 2>/dev/null | wc -l)" -gt 0 ]; then
        python3 create_single.py "unified/$TARGET_EXPERIMENT" "unified/$TARGET_EXPERIMENT/all_data_${TARGET_EXPERIMENT}.csv"
        echo "Processed experiment: $TARGET_EXPERIMENT"
    else
        echo "No CSV files found for experiment: $TARGET_EXPERIMENT, skipping..."
        rmdir "unified/$TARGET_EXPERIMENT" 2>/dev/null
    fi
    
    echo $(pwd)
done