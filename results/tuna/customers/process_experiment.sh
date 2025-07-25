#!/bin/bash

echo "🚀 UNIFIED METRICS COMBINER"
echo "Current directory: $(pwd)"
echo "=========================="

# Updated experiments list
EXPERIMENTS=("baseline" "cpu_stress" "IOpressure" "net_loss" "mem_stress")

# Function to copy files from cleaned_data directory if it exists
copy_cleaned_files() {
    local source_dir=$1
    local target_dir=$2
    local desc=$3
    
    if [ -d "$source_dir" ]; then
        echo "  📂 Copying from: $source_dir ($desc)"
        find "$source_dir" -type f -name "*.csv" -exec cp -f {} "$target_dir/" \;
        local count=$(find "$source_dir" -name "*.csv" 2>/dev/null | wc -l)
        echo "     Copied $count CSV files"
    else
        echo "  ❌ Directory not found: $source_dir"
    fi
}

# Process each experiment
for TARGET_EXPERIMENT in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "🔧 Processing experiment: $TARGET_EXPERIMENT"
    echo "--------------------------------------------"
    
    # Create target directory
    mkdir -p "unified/$TARGET_EXPERIMENT"
    BASE_DIR=$(pwd)
    
    echo "  📋 Gathering files from cleaned_data directories..."
    
    # Copy ONLY from cleaned_data directories (should now include ALL data after TUNA export)
    copy_cleaned_files "cpu related/cleaned_data/$TARGET_EXPERIMENT" "unified/$TARGET_EXPERIMENT" "CPU cleaned"
    copy_cleaned_files "IO related/cleaned_data/$TARGET_EXPERIMENT" "unified/$TARGET_EXPERIMENT" "I/O cleaned"
    copy_cleaned_files "memory related/cleaned_data/$TARGET_EXPERIMENT" "unified/$TARGET_EXPERIMENT" "Memory cleaned"
    copy_cleaned_files "tcp related/cleaned_data/$TARGET_EXPERIMENT" "unified/$TARGET_EXPERIMENT" "TCP cleaned (includes all TCP data)"
    
    # Check total files collected
    total_files=$(find "unified/$TARGET_EXPERIMENT" -name "*.csv" 2>/dev/null | wc -l)
    echo "  📊 Total files collected: $total_files"
    
    if [ "$total_files" -gt 0 ]; then
        echo "  🔧 Running Python combiner..."
        python3 create_single.py "unified/$TARGET_EXPERIMENT"
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Successfully processed: $TARGET_EXPERIMENT"
        else
            echo "  ❌ Failed to process: $TARGET_EXPERIMENT"
        fi
    else
        echo "  ⚠️  No CSV files found for experiment: $TARGET_EXPERIMENT"
    fi
    
    echo "  📍 Current directory: $(pwd)"
done

echo ""
echo "🎉 All experiments processed!"
echo ""
echo "📊 PROCESSING SUMMARY:"
echo "===================="
for exp in "${EXPERIMENTS[@]}"; do
    if [ -d "unified/$exp" ]; then
        combined_file="unified/$exp/all_metrics_combined_$exp.csv"
        if [ -f "$combined_file" ]; then
            lines=$(wc -l < "$combined_file" 2>/dev/null || echo "0")
            cols=$(head -1 "$combined_file" 2>/dev/null | tr ',' '\n' | wc -l)
            echo "✅ $exp: $lines rows, $cols columns"
            
            # Check if TCP data is included (should include apigateway now)
            if grep -q "tcp_apigateway\|apigateway" "$combined_file" 2>/dev/null; then
                echo "   🌐 Includes TCP ApiGateway data"
            else
                echo "   ⚠️  No ApiGateway data detected"
            fi
        else
            echo "❌ $exp: No combined file created"
        fi
    else
        echo "❌ $exp: Directory not found"
    fi
done

echo ""
echo "📁 Results are in the 'unified/' directory"