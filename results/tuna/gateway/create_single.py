import pandas as pd
import glob
import os
import sys
from collections import defaultdict

target_dir = sys.argv[1]
output_file = sys.argv[2]

csv_files = glob.glob(os.path.join(target_dir, "*.csv"))

METRIC_GROUPS = {
    'memory': ['memcache', 'memutil', 'memavailable'],
    'cpu': ['cpu_iowait', 'cpu_irq', 'cpu_system_msec', 'cpu_user_msec', 'cpu_util_per'],
    'IO': ['block_count_latency_device', 'read_bytes', 'write_bytes'],
    'network': ['apigateway', 'customersservice', 'srtt', 'vetsservice', 'visitsservice',]
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

print(f"Found {len(csv_files)} CSV files")
print(f"Grouped into {len(grouped_files)} metric types:")
for group, files in grouped_files.items():
    print(f"  {group}: {len(files)} files")

# Crea output_dir se non esiste
output_dir = os.path.dirname(output_file)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Processa ciascun gruppo
merged_data = {}

for metric_group, files in grouped_files.items():
    print(f"\nProcessing {metric_group} group with {len(files)} files:")
    
    dfs = []
    time_column = None
    
    for file in sorted(files):
        print(f"  Reading: {os.path.basename(file)}")
        
        try:
            df = pd.read_csv(file)
            
            # Gestione della colonna time
            if "time" in df.columns:
                if time_column is None:
                    time_column = df["time"].copy()
                # Rimuovi la colonna time da questo DataFrame
                df = df.drop(columns=["time"])
            
            # Aggiungi prefisso basato sul nome del file
            prefix = os.path.splitext(os.path.basename(file))[0]
            df = df.add_prefix(f"{prefix}_")
            
            dfs.append(df)
            
        except Exception as e:
            print(f"    Error reading {file}: {e}")
            continue
    
    if dfs:
        # Unisci tutti i DataFrame del gruppo
        merged = pd.concat(dfs, axis=1)
        
        # Aggiungi la colonna time all'inizio se presente
        if time_column is not None:
            merged.insert(0, 'time', time_column)
        
        merged_data[metric_group] = merged
        print(f"  Merged shape: {merged.shape}")

# Ora unisci tutti i gruppi in un singolo DataFrame
print(f"\nCombining all {len(merged_data)} metric groups...")

all_dfs = []
final_time_column = None

for group_name, group_df in merged_data.items():
    print(f"  Adding {group_name} group with {group_df.shape[1]} columns")
    
    # Gestione della colonna time
    if "time" in group_df.columns:
        if final_time_column is None:
            final_time_column = group_df["time"].copy()
        # Rimuovi la colonna time da questo gruppo
        group_df = group_df.drop(columns=["time"])
    
    # Aggiungi prefisso del gruppo alle colonne
    group_df = group_df.add_prefix(f"{group_name}_")
    all_dfs.append(group_df)

# Unisci tutti i gruppi
if all_dfs:
    final_merged = pd.concat(all_dfs, axis=1)
    
    # Aggiungi la colonna time all'inizio
    if final_time_column is not None:
        final_merged.insert(0, 'time', final_time_column)
    
    # Salva il file finale
    final_merged.to_csv(output_file, index=False)
    
    print(f"\n✅ File finale salvato: {output_file}")
    print(f"   Dimensioni finali: {final_merged.shape}")
    print(f"   Colonne per gruppo:")
    
    # Mostra statistiche finali
    for group in merged_data.keys():
        group_cols = [col for col in final_merged.columns if col.startswith(f"{group}_")]
        print(f"     {group}: {len(group_cols)} colonne")
else:
    print("❌ Nessun dato da unire")

print("\n✅ Unione completata.")