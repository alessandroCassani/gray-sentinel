import pandas as pd
import glob
import os
import sys
from collections import defaultdict

target_dir = sys.argv[1]

csv_files = [f for f in glob.glob(os.path.join(target_dir, "*.csv")) 
             if not os.path.basename(f).startswith("all_")]

# Updated metric groups based on your file structure
METRIC_GROUPS = {
    'memory': ['memcache', 'memutil', 'memavailable'],
    'cpu': ['iowait', 'irq', 'system', 'user', 'utilization'],
    'IO': ['blocklatency', 'readbytes', 'writebytes'],
    'network': ['api_gateway', 'customers_service', 'vets_service', 'visits_service', 'srtt'],
    'tcp': ['retrans', 'packets']
}

def get_metric_group(filename):
    filename_lower = filename.lower()
    for group, keywords in METRIC_GROUPS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return group
    
    # Fallback per file non categorizzati
    return 'other'

grouped_files = defaultdict(list)

for file in csv_files:
    filename = os.path.basename(file)
    metric_group = get_metric_group(filename)
    if metric_group:  # Solo se il gruppo è stato identificato
        grouped_files[metric_group].append(file)

for metric_group, files in grouped_files.items():
    dfs = []
    time_column = None
    
    for file in sorted(files):
        try:
            df = pd.read_csv(file)
            
            # Try to find time column
            if time_column is None:
                time_cols = [col for col in df.columns if col.lower() in ['minutes', 'time', 'timestamp']]
                if time_cols:
                    time_column = df[time_cols[0]].copy()
            
            # Drop time columns from this dataframe
            cols_to_drop = [col for col in df.columns if col.lower() in ['time', 'timestamp', 'minutes']]
            df = df.drop(columns=cols_to_drop, errors='ignore')
            
            # Get base filename without extension
            filename_base = os.path.splitext(os.path.basename(file))[0]
            
            # Handle different naming patterns
            if metric_group == 'memory':
                # For memory metrics, use the base name
                metric_name = filename_base.split('_')[0] if '_' in filename_base else filename_base
                if len(df.columns) == 1:
                    df.columns = [metric_name]
                else:
                    df = df.add_prefix(f"{metric_name}_")
            else:
                # For other metrics, use full filename as prefix
                metric_name = filename_base.replace('.', '_')
                df = df.add_prefix(f"{metric_name}_")
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Warning: Could not process {file}: {e}")
            continue
    
    if dfs:
        merged = pd.concat(dfs, axis=1)
        
        if time_column is not None:
            merged.insert(0, 'minutes', time_column)
        
        output_file = os.path.join(target_dir, f"all_{metric_group}.csv")
        merged.to_csv(output_file, index=False)
        print(f"Created: {output_file}")

# Combine all metric group files into final combined file
all_files = glob.glob(os.path.join(target_dir, "all_*.csv"))

if all_files:
    final_dfs = []
    final_time_column = None
    
    for file in sorted(all_files):
        try:
            df = pd.read_csv(file)
            
            if final_time_column is None and 'minutes' in df.columns:
                final_time_column = df['minutes'].copy()
            
            if 'minutes' in df.columns:
                df = df.drop(columns=['minutes'])
            
            filename = os.path.basename(file)
            metric_type = filename.replace("all_", "").replace(".csv", "")
            df = df.add_prefix(f"{metric_type}_")
            
            final_dfs.append(df)
            
        except Exception as e:
            print(f"Warning: Could not process {file}: {e}")
            continue
    
    if final_dfs:
        final_merged = pd.concat(final_dfs, axis=1)
        
        if final_time_column is not None:
            final_merged.insert(0, 'minutes', final_time_column)
        
        experiment_name = os.path.basename(target_dir)
        final_output = os.path.join(target_dir, f"all_metrics_combined_{experiment_name}.csv")
        
        final_merged.to_csv(final_output, index=False)
        print(f"Final combined file created: {final_output}")
    else:
        print("No data files found to combine")
else:
    print("No all_*.csv files found")