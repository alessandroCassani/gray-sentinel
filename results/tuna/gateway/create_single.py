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

print(f"Found {len(csv_files)} CSV files (excluding existing all_* files)")
print(f"Grouped into {len(grouped_files)} metric types:")
for group, files in grouped_files.items():
    print(f"  {group}: {len(files)} files")

print("\nRemoving old all_*.csv and combined files...")
old_files = glob.glob(os.path.join(target_dir, "all_*.csv"))
combined_files = glob.glob(os.path.join(target_dir, "all_metrics_combined*.csv"))
for old_file in old_files + combined_files:
    os.remove(old_file)
    print(f"  Removed: {os.path.basename(old_file)}")

# STEP 1: Crea i file all_*.csv per ogni gruppo di metriche
for metric_group, files in grouped_files.items():
    print(f"\nProcessing {metric_group} group with {len(files)} files:")
    
    dfs = []
    time_column = None
    
    for file in sorted(files):
        print(f"  Reading: {os.path.basename(file)}")
        
        try:
            df = pd.read_csv(file)
            
            if time_column is None:
                time_cols = [col for col in df.columns if col.lower() in ['time', 'timestamp', 'minutes']]
                if time_cols:
                    time_column = df[time_cols[0]].copy()
            
            cols_to_drop = []
            for col in df.columns:
                if col.lower() in ['source', 'minutes', 'timestamp', 'time']:
                    cols_to_drop.append(col)
            df = df.drop(columns=cols_to_drop, errors='ignore')
            
            filename_base = os.path.splitext(os.path.basename(file))[0]
            
            if metric_group == 'memory':
                metric_name = filename_base.split('_')[0]  # es. "memavailable"
                if len(df.columns) == 1:
                    df.columns = [metric_name]
                else:
                    df = df.add_prefix(f"{metric_name}_")
            else:
                metric_name = filename_base.split('_')[0]
                df = df.add_prefix(f"{metric_name}_")
            
            dfs.append(df)
            
        except Exception as e:
            print(f"    Error reading {file}: {e}")
            continue
    
    if dfs:
        merged = pd.concat(dfs, axis=1)
        
        if time_column is not None:
            merged.insert(0, 'time', time_column)
        
        output_file = os.path.join(target_dir, f"all_{metric_group}.csv")
        merged.to_csv(output_file, index=False)
        
        print(f"  Saved: {output_file}")
        print(f"  Shape: {merged.shape}")
        
        cols_to_show = list(merged.columns[:8])
        print(f"  First columns: {cols_to_show}")

print(f"\n✅ Created individual all_*.csv files")

# STEP 2: Unisci tutti i file all_*.csv in un singolo file
print(f"\n🔗 STEP 2: Merging all_*.csv files into single combined file...")

# Trova tutti i file all_*.csv appena creati
all_files = glob.glob(os.path.join(target_dir, "all_*.csv"))

if not all_files:
    print("❌ No all_*.csv files found to merge")
else:
    print(f"📁 Found {len(all_files)} files to merge:")
    for file in sorted(all_files):
        print(f"  📄 {os.path.basename(file)}")
    
    # Lista per contenere tutti i DataFrame
    final_dfs = []
    final_time_column = None
    
    print(f"\n🔄 Processing files for final merge...")
    
    for file in sorted(all_files):
        filename = os.path.basename(file)
        print(f"  📖 Reading: {filename}")
        
        try:
            df = pd.read_csv(file)
            
            # Salva la colonna time dal primo file
            if final_time_column is None and 'time' in df.columns:
                final_time_column = df['time'].copy()
                print(f"    ⏰ Using time column from {filename}")
            
            # Rimuovi la colonna time da questo DataFrame
            if 'time' in df.columns:
                df = df.drop(columns=['time'])
            
            # Aggiungi prefisso basato sul nome del file (rimuovi "all_" e ".csv")
            metric_type = filename.replace("all_", "").replace(".csv", "")
            df = df.add_prefix(f"{metric_type}_")
            
            final_dfs.append(df)
            print(f"    ✅ Added {len(df.columns)} columns with prefix '{metric_type}_'")
            
        except Exception as e:
            print(f"    ❌ Error reading {filename}: {e}")
            continue
    
    if final_dfs:
        print(f"\n🔗 Merging {len(final_dfs)} DataFrames into final combined file...")
        
        # Unisci tutti i DataFrame
        final_merged = pd.concat(final_dfs, axis=1)
        
        # Aggiungi la colonna time all'inizio se presente
        if final_time_column is not None:
            final_merged.insert(0, 'time', final_time_column)
            print(f"    ⏰ Added time column as first column")
        
        # Nome del file finale
        experiment_name = os.path.basename(target_dir)
        final_output = os.path.join(target_dir, f"all_metrics_combined_{experiment_name}.csv")
        
        # Salva il file combinato finale
        final_merged.to_csv(final_output, index=False)
        
        print(f"\n🎉 FINAL COMBINED FILE CREATED!")
        print(f"📄 File: {os.path.basename(final_output)}")
        print(f"📊 Shape: {final_merged.shape}")
        print(f"📋 Total columns: {len(final_merged.columns)}")
        
        # Mostra le prime colonne per verifica
        first_cols = list(final_merged.columns[:10])
        print(f"🔍 First 10 columns: {first_cols}")
        
        if len(final_merged.columns) > 10:
            print(f"   ... and {len(final_merged.columns) - 10} more columns")
        
        # Mostra struttura per gruppo
        print(f"\n📊 Columns by metric group:")
        for group in ['memory', 'cpu', 'IO', 'network']:
            group_cols = [col for col in final_merged.columns if col.startswith(f"{group}_")]
            if group_cols:
                print(f"  {group}: {len(group_cols)} columns")
    
    else:
        print("❌ No valid data to merge into final file")

print(f"\n📁 Final files in {target_dir}:")
final_files = glob.glob(os.path.join(target_dir, "all_*.csv"))
for f in sorted(final_files):
    file_size = os.path.getsize(f) / 1024  # KB
    print(f"  📄 {os.path.basename(f)} ({file_size:.1f} KB)")