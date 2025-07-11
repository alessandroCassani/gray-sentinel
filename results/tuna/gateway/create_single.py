import pandas as pd
import glob
import os
import sys
from collections import defaultdict

target_dir = sys.argv[1]

# Trova tutti i CSV che NON iniziano con "all_"
csv_files = [f for f in glob.glob(os.path.join(target_dir, "*.csv")) 
             if not os.path.basename(f).startswith("all_")]

METRIC_GROUPS = {
    'memory': ['memcache', 'memutil', 'memavailable'],
    'cpu': ['iowait', 'irq', 'system', 'user', 'utilization'],
    'IO': ['blocklatency', 'readbytes', 'writebytes'],
    'network': ['apigateway', 'customersservice', 'srtt', 'vetsservice', 'visitsservice']
}

def get_metric_group(filename):
    """Determina il gruppo di metriche basato sul nome del file."""
    filename_lower = filename.lower()
    
    for group, keywords in METRIC_GROUPS.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return group
    
    # Se non trova un gruppo, usa il primo token del nome file
    base = os.path.splitext(filename)[0]
    return base.split('_')[0]

# Raggruppa i file per tipo di metrica
grouped_files = defaultdict(list)

for file in csv_files:
    filename = os.path.basename(file)
    metric_group = get_metric_group(filename)
    grouped_files[metric_group].append(file)

print(f"Found {len(csv_files)} CSV files (excluding existing all_* files)")
print(f"Grouped into {len(grouped_files)} metric types:")
for group, files in grouped_files.items():
    print(f"  {group}: {len(files)} files")

# Prima rimuovi i vecchi file all_*.csv
print("\nRemoving old all_*.csv files...")
old_files = glob.glob(os.path.join(target_dir, "all_*.csv"))
for old_file in old_files:
    os.remove(old_file)
    print(f"  Removed: {os.path.basename(old_file)}")

# Processa ciascun gruppo e crea un file separato per metrica
for metric_group, files in grouped_files.items():
    print(f"\nProcessing {metric_group} group with {len(files)} files:")
    
    dfs = []
    source_column = None
    minutes_column = None
    timestamp_column = None
    
    for file in sorted(files):
        print(f"  Reading: {os.path.basename(file)}")
        
        try:
            df = pd.read_csv(file)
            
            # Salva le colonne comuni (solo dal primo file)
            if source_column is None and 'source' in df.columns:
                source_column = df['source'].copy()
            if minutes_column is None and 'minutes' in df.columns:
                minutes_column = df['minutes'].copy()
            if timestamp_column is None and 'Timestamp' in df.columns:
                timestamp_column = df['Timestamp'].copy()
            
            # Rimuovi le colonne comuni da questo DataFrame
            cols_to_drop = []
            for col in df.columns:
                if col.lower() in ['source', 'minutes', 'timestamp']:
                    cols_to_drop.append(col)
            df = df.drop(columns=cols_to_drop, errors='ignore')
            
            # Estrai il nome della metrica dal nome del file
            filename_base = os.path.splitext(os.path.basename(file))[0]
            
            # Aggiungi il prefisso della metrica a tutte le colonne
            metric_name = filename_base.split('_')[0]
            df = df.add_prefix(f"{metric_name}_")
            
            dfs.append(df)
            
        except Exception as e:
            print(f"    Error reading {file}: {e}")
            continue
    
    if dfs:
        # Unisci tutti i DataFrame del gruppo
        merged = pd.concat(dfs, axis=1)
        
        # Aggiungi le colonne comuni all'inizio
        col_index = 0
        if source_column is not None:
            merged.insert(col_index, 'source', source_column)
            col_index += 1
        if minutes_column is not None:
            merged.insert(col_index, 'minutes', minutes_column)
            col_index += 1
        if timestamp_column is not None:
            merged.insert(col_index, 'Timestamp', timestamp_column)
        
        # Salva il file per questa metrica
        output_file = os.path.join(target_dir, f"all_{metric_group}.csv")
        merged.to_csv(output_file, index=False)
        
        print(f"  Saved: {output_file}")
        print(f"  Shape: {merged.shape}")
        
        # Mostra le prime colonne per debug
        cols_to_show = list(merged.columns[:8])
        print(f"  First columns: {cols_to_show}")

print(f"\n✅ Creati file all_*.csv per ogni gruppo di metriche in: {target_dir}")
print("\nFinal files created:")
final_files = glob.glob(os.path.join(target_dir, "all_*.csv"))
for f in final_files:
    print(f"  {os.path.basename(f)}")