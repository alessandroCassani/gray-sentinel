import pandas as pd
import glob
import os
import sys
from collections import defaultdict

target_dir = sys.argv[1]

csv_files = [f for f in glob.glob(os.path.join(target_dir, "*.csv")) 
             if not os.path.basename(f).startswith("all_")]

METRIC_GROUPS = {
    'memory': ['memcache', 'memutil', 'memavailable'],
    'cpu': ['iowait', 'irq', 'system', 'user', 'utilization'],
    'IO': ['blocklatency', 'readbytes', 'writebytes'],
    'network': ['apigateway', 'customersservice', 'srtt', 'vetsservice', 'visitsservice']
}

def get_metric_group(filename):
    filename_lower = filename.lower()
    for group, keywords in METRIC_GROUPS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return group

grouped_files = defaultdict(list)

for file in csv_files:
    filename = os.path.basename(file)
    metric_group = get_metric_group(filename)
    grouped_files[metric_group].append(file)

old_files = glob.glob(os.path.join(target_dir, "all_*.csv"))
combined_files = glob.glob(os.path.join(target_dir, "all_metrics_combined*.csv"))

for old_file in old_files + combined_files:
    os.remove(old_file)

for metric_group, files in grouped_files.items():
    dfs = []
    time_column = None
    
    for file in sorted(files):
        try:
            df = pd.read_csv(file)
            
            if time_column is None:
                minutes_cols = [col for col in df.columns if col.lower() == 'minutes']
                if minutes_cols:
                    time_column = df[minutes_cols[0]].copy()
            
            cols_to_drop = []
            for col in df.columns:
                if col.lower() in ['time', 'timestamp']:
                    cols_to_drop.append(col)

            df = df.drop(columns=cols_to_drop, errors='ignore')
            
            filename_base = os.path.splitext(os.path.basename(file))[0]
            
            if metric_group == 'memory':
                metric_name = filename_base.split('_')[0]
                if len(df.columns) == 1:
                    df.columns = [metric_name]
                else:
                    df = df.add_prefix(f"{metric_name}_")
            else:
                metric_name = filename_base.split('_')[0]
                df = df.add_prefix(f"{metric_name}_")
            
            dfs.append(df)
            
        except Exception as e:
            continue
    
    if dfs:
        merged = pd.concat(dfs, axis=1)
        
        if time_column is not None:
            merged.insert(0, 'minutes', time_column)
        
        output_file = os.path.join(target_dir, f"all_{metric_group}.csv")
        merged.to_csv(output_file, index=False)

all_files = glob.glob(os.path.join(target_dir, "all_*.csv"))

if all_files:
    final_dfs = []
    final_time_column = None
    
    for file in sorted(all_files):
        filename = os.path.basename(file)
        
        try:
            df = pd.read_csv(file)
            
            if final_time_column is None and 'minutes' in df.columns:
                final_time_column = df['minutes'].copy()
            
            if 'minutes' in df.columns:
                df = df.drop(columns=['minutes'])
            
            metric_type = filename.replace("all_", "").replace(".csv", "")
            df = df.add_prefix(f"{metric_type}_")
            
            final_dfs.append(df)
            
        except Exception as e:
            continue
    
    if final_dfs:
        final_merged = pd.concat(final_dfs, axis=1)
        
        if final_time_column is not None:
            final_merged.insert(0, 'minutes', final_time_column)
        
        experiment_name = os.path.basename(target_dir)
        final_output = os.path.join(target_dir, f"all_metrics_combined_{experiment_name}.csv")
        
        final_merged.to_csv(final_output, index=False)