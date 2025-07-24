#!/bin/bash
echo $(pwd)

EXPERIMENTS=("cache filling" "oom" "threadfull")

for TARGET_EXPERIMENT in "${EXPERIMENTS[@]}"; do
    # Clean up any existing directory first
    if [ -d "unified/$TARGET_EXPERIMENT" ]; then
        rm -rf "unified/$TARGET_EXPERIMENT"
    fi
    mkdir -p "unified/$TARGET_EXPERIMENT"
    BASE_DIR=$(pwd)
    
    if [ -d "cpu related/cleaned_data/$TARGET_EXPERIMENT" ]; then
        find "cpu related/cleaned_data/$TARGET_EXPERIMENT" -type f -name "*.csv" -exec cp {} "unified/$TARGET_EXPERIMENT/" \;
    fi

    if [ -d "IO related/cleaned_data/$TARGET_EXPERIMENT" ]; then
        find "IO related/cleaned_data/$TARGET_EXPERIMENT" -type f -name "*.csv" -exec cp {} "unified/$TARGET_EXPERIMENT/" \;
    fi

    if [ -d "memory related/cleaned_data/$TARGET_EXPERIMENT" ]; then
        find "memory related/cleaned_data/$TARGET_EXPERIMENT" -type f -name "*.csv" -exec cp {} "unified/$TARGET_EXPERIMENT/" \;
    fi

    if [ -d "tcp related/cleaned_data/$TARGET_EXPERIMENT" ]; then
        find "tcp related/cleaned_data/$TARGET_EXPERIMENT" -type f -name "*.csv" -exec cp {} "unified/$TARGET_EXPERIMENT/" \;
    fi

    if [ "$(find "unified/$TARGET_EXPERIMENT" -name "*.csv" 2>/dev/null | wc -l)" -gt 0 ]; then
        python3 create_single.py "unified/$TARGET_EXPERIMENT" "unified/$TARGET_EXPERIMENT/all_data_${TARGET_EXPERIMENT}.csv"
        echo "Processed experiment: $TARGET_EXPERIMENT"
    else
        echo "No CSV files found for experiment: $TARGET_EXPERIMENT, skipping..."
        rmdir "unified/$TARGET_EXPERIMENT" 2>/dev/null
    fi
    
    echo $(pwd)
done