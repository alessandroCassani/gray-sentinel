#!/bin/bash

cho "🚀 SIMPLE CSV COMBINER"
echo "====================="
echo "Current directory: $(pwd)"
echo ""

# Your experiments
EXPERIMENTS=("baseline" "cpu_stress" "mem_stress" "IOpressure" "net_loss")

# Metric source directories (the main directories in your structure)
METRIC_DIRS=("cpu related" "memory related" "IO related" "tcp related")

echo "🔍 Checking available metric directories:"
for metric_dir in "${METRIC_DIRS[@]}"; do
    cleaned_path="$metric_dir/cleaned_data"
    if [ -d "$cleaned_path" ]; then
        echo "  ✅ Found: $cleaned_path/"
        # Show available experiments in this metric dir
        if [ "$(ls -1 "$cleaned_path" 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "    Experiments: $(ls -1 "$cleaned_path" | tr '\\n' ' ')"
        fi
    else
        echo "  ❌ Not found: $cleaned_path/"
    fi
done
echo ""

# Function to copy files from metric directories
copy_metric_files() {
    local experiment=$1
    local target_dir=$2
    
    echo "    📋 Gathering cleaned files for $experiment:"
    
    for metric_dir in "${METRIC_DIRS[@]}"; do
        source_path="$metric_dir/cleaned_data/$experiment"
        
        if [ -d "$source_path" ]; then
            csv_count=$(find "$source_path" -maxdepth 1 -name "*.csv" | wc -l)
            if [ "$csv_count" -gt 0 ]; then
                cp "$source_path"/*.csv "$target_dir/" 2>/dev/null
                echo "      ✓ $metric_dir: $csv_count files"
            else
                echo "      - $metric_dir: no CSV files"
            fi
        else
            echo "      - $metric_dir: directory not found"
        fi
    done
}

# Process each experiment
for experiment in "${EXPERIMENTS[@]}"; do
    echo "🔧 Processing experiment: $experiment"
    echo "--------------------------------"
    
    target_dir="unified/$experiment"
    
    # Create target directory
    mkdir -p "$target_dir"
    
    # Copy files from all metric directories
    copy_metric_files "$experiment" "$target_dir"
    
    # Check if any files were copied
    total_files=$(find "$target_dir" -maxdepth 1 -name "*.csv" | wc -l)
    
    if [ "$total_files" -gt 0 ]; then
        echo "    📊 Total files collected: $total_files"
        
        # Run combine script on target directory
        if [ -f "simple_combine.py" ]; then
            echo "    🔧 Running combiner..."
            python3 simple_combine.py "$target_dir"
        else
            echo "    ⚠️  simple_combine.py not found in current directory"
        fi
    else
        echo "    ❌ No CSV files found for $experiment"
        rmdir "$target_dir" 2>/dev/null
    fi
    
    echo ""
done

echo "🎉 All done!"

# Show results
echo ""
echo "📊 RESULTS:"
echo "==========="
for experiment in "${EXPERIMENTS[@]}"; do
    file="unified/$experiment/all_metrics_combined_$experiment.csv"
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        cols=$(head -1 "$file" | tr ',' '\\n' | wc -l)
        echo "  ✅ $experiment: $lines rows, $cols columns"
    else
        echo "  ❌ $experiment: not created"
    fi
done

echo ""
echo "📁 All combined files are in the 'unified/' directory:"
find unified/ -name "all_metrics_combined_*.csv" 2>/dev/null | sort

echo ""
echo "🗂️  Final structure created:"
if [ -d "unified" ]; then
    for experiment in "${EXPERIMENTS[@]}"; do
        if [ -d "unified/$experiment" ]; then
            echo "unified/$experiment/:"
            ls -1 "unified/$experiment/"*.csv 2>/dev/null | head -5 | sed 's/^/  /'
            total=$(ls -1 "unified/$experiment/"*.csv 2>/dev/null | wc -l)
            if [ "$total" -gt 5 ]; then
                echo "  ... and $((total - 5)) more files"
            fi
        fi
    done
fi