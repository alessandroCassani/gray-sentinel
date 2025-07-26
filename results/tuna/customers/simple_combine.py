import pandas as pd
import glob
import os
import sys
from pathlib import Path

def combine_csvs_simple(target_dir):
    """Combine all CSV files in directory into one file"""
    
    print(f"📂 Processing: {target_dir}")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("all_")]
    
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    print(f"📋 Found {len(csv_files)} files")
    
    dfs = []
    time_column = None
    
    # Process each file
    for file_path in sorted(csv_files):
        try:
            df = pd.read_csv(file_path)
            filename = Path(file_path).stem
            
            print(f"  ✓ {filename}: {df.shape}")
            
            # Get time column from first file
            if time_column is None:
                time_cols = [col for col in df.columns if col.lower() in ['minutes', 'time']]
                if time_cols:
                    time_column = df[time_cols[0]].copy()
            
            # Remove time columns from this df
            df = df.drop(columns=[col for col in df.columns if col.lower() in ['minutes', 'time']], errors='ignore')
            
            # Add filename prefix to avoid column conflicts
            df = df.add_prefix(f"{filename}_")
            
            dfs.append(df)
            
        except Exception as e:
            print(f"  ❌ Error with {filename}: {e}")
    
    # Combine all dataframes
    if dfs:
        combined = pd.concat(dfs, axis=1)
        
        # Add time column at start
        if time_column is not None:
            combined.insert(0, 'minutes', time_column)
        
        # Save combined file
        experiment_name = os.path.basename(target_dir)
        output_file = os.path.join(target_dir, f"all_metrics_combined_{experiment_name}.csv")
        combined.to_csv(output_file, index=False)
        
        print(f"✅ Created: {output_file}")
        print(f"   Shape: {combined.shape}")
        return True
    
    return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 simple_combine.py <directory>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    
    if not os.path.exists(target_dir):
        print(f"❌ Directory not found: {target_dir}")
        sys.exit(1)
    
    success = combine_csvs_simple(target_dir)
    
    if success:
        print("🎉 Done!")
    else:
        print("❌ Failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()