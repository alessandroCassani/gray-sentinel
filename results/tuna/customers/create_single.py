#!/usr/bin/env python3

import pandas as pd
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 create_single.py <target_directory>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    
    if not os.path.exists(target_dir):
        print(f"❌ Directory does not exist: {target_dir}")
        sys.exit(1)
    
    print(f"🔧 Processing directory: {target_dir}")
    
    # Find all CSV files (exclude already combined files)
    csv_files = [f for f in glob.glob(os.path.join(target_dir, "*.csv")) 
                 if not os.path.basename(f).startswith("all_")]
    
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    print(f"📂 Found {len(csv_files)} CSV files")
    
    # Original metric groups (simplified - TUNA should have exported all TCP data to cleaned_data)
    METRIC_GROUPS = {
        'memory': ['memcache', 'memutil', 'memavailable'],
        'cpu': ['iowait', 'irq', 'system', 'user', 'utilization'],
        'io': ['blocklatency', 'readbytes', 'writebytes'],
        'network': ['apigateway', 'customersservice', 'srtt', 'vetsservice', 'visitsservice', 'tcp']
    }
    
    def get_metric_group(filename):
        filename_lower = filename.lower()
        for group, keywords in METRIC_GROUPS.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return group
        return 'other'
    
    # Group files by metric type
    grouped_files = defaultdict(list)
    
    for file in csv_files:
        filename = os.path.basename(file)
        metric_group = get_metric_group(filename)
        grouped_files[metric_group].append(file)
        print(f"  {filename} → {metric_group}")
    
    print(f"\n📊 Grouped into {len(grouped_files)} metric categories:")
    for group, files in grouped_files.items():
        print(f"  {group}: {len(files)} files")
    
    # Clean up old combined files
    old_files = glob.glob(os.path.join(target_dir, "all_*.csv"))
    combined_files = glob.glob(os.path.join(target_dir, "all_metrics_combined*.csv"))
    
    for old_file in old_files + combined_files:
        os.remove(old_file)
        print(f"🗑️  Removed old file: {os.path.basename(old_file)}")
    
    # Process each metric group
    for metric_group, files in grouped_files.items():
        print(f"\n🔧 Processing {metric_group} group...")
        
        dfs = []
        time_column = None
        
        for file in sorted(files):
            try:
                df = pd.read_csv(file)
                filename_base = os.path.splitext(os.path.basename(file))[0]
                
                print(f"  ✓ {filename_base}: {df.shape}")
                
                # Extract time column from first file
                if time_column is None:
                    time_cols = [col for col in df.columns if col.lower() in ['minutes', 'time']]
                    if time_cols:
                        time_column = df[time_cols[0]].copy()
                
                # Remove time-related columns from this dataframe
                df = df.drop(columns=[col for col in df.columns if col.lower() in ['time', 'timestamp', 'minutes']], errors='ignore')
                
                # Add filename prefix to avoid column conflicts
                if metric_group == 'memory':
                    # Special handling for memory - use shorter prefixes
                    metric_name = filename_base.split('_')[0] if '_' in filename_base else filename_base
                    if len(df.columns) == 1:
                        df.columns = [f"{metric_name}_{df.columns[0]}"]
                    else:
                        df = df.add_prefix(f"{metric_name}_")
                else:
                    # For all other groups, use full filename as prefix
                    df = df.add_prefix(f"{filename_base}_")
                
                dfs.append(df)
                
            except Exception as e:
                print(f"  ❌ Error processing {filename_base}: {e}")
                continue
        
        # Combine all dataframes in this group
        if dfs:
            merged = pd.concat(dfs, axis=1)
            
            # Add time column at the beginning
            if time_column is not None:
                merged.insert(0, 'minutes', time_column)
            
            # Save group file
            output_file = os.path.join(target_dir, f"all_{metric_group}.csv")
            merged.to_csv(output_file, index=False)
            print(f"  💾 Saved: all_{metric_group}.csv ({merged.shape})")
    
    # Combine all group files into final file
    print(f"\n🔗 Creating final combined file...")
    all_files = glob.glob(os.path.join(target_dir, "all_*.csv"))
    
    if all_files:
        final_dfs = []
        final_time_column = None
        
        for file in sorted(all_files):
            filename = os.path.basename(file)
            
            try:
                df = pd.read_csv(file)
                
                # Get time column from first file
                if final_time_column is None and 'minutes' in df.columns:
                    final_time_column = df['minutes'].copy()
                
                # Remove minutes column from this dataframe
                if 'minutes' in df.columns:
                    df = df.drop(columns=['minutes'])
                
                # Add metric group prefix
                metric_type = filename.replace("all_", "").replace(".csv", "")
                df = df.add_prefix(f"{metric_type}_")
                
                final_dfs.append(df)
                print(f"  ✓ Added {metric_type}: {df.shape}")
                
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
                continue
        
        # Create final combined dataframe
        if final_dfs:
            final_merged = pd.concat(final_dfs, axis=1)
            
            # Add time column at the beginning
            if final_time_column is not None:
                final_merged.insert(0, 'minutes', final_time_column)
            
            # Save final combined file
            experiment_name = os.path.basename(target_dir)
            final_output = os.path.join(target_dir, f"all_metrics_combined_{experiment_name}.csv")
            
            final_merged.to_csv(final_output, index=False)
            print(f"\n✅ Final combined file created: {final_output}")
            print(f"   Shape: {final_merged.shape}")
            
            # Show sample of column names
            print(f"   Sample columns: {list(final_merged.columns)[:10]}...")
            
            return True
        else:
            print("❌ No data to combine")
            return False
    else:
        print("❌ No group files found")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 Processing completed successfully!")
    else:
        print("❌ Processing failed!")
        sys.exit(1)