"""
TUNA Data Processor for Failure Detection Pipeline
Processes all datasets through TUNA cleaning and exports cleaned CSVs for ML training
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import TUNA modules
from data_utils import (extract_values_by_type, detect_metric_type, validate_datasets)
from ml_training import train_models_for_datasets
from tuna_core import apply_tuna_to_single_series, print_tuna_summary


class TunaDataProcessor:
    """Processes datasets through TUNA pipeline and exports cleaned data for ML training"""
    
    def __init__(self, penalty_factor: float = 0.75, random_state: int = 42, n_jobs: int = -1):
        self.penalty_factor = penalty_factor
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trained_models = {}
        self.tuna_results = {}
        self.processed_datasets = {}
        
    def load_all_datasets(self, data_path: str, file_pattern: str = "*.csv") -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all datasets from directory structure
        
        Expected structure:
        data_path/
        ├── metric1/
        │   ├── baseline.csv
        │   ├── experiment1.csv
        │   └── experiment2.csv
        ├── metric2/
        │   └── ...
        
        Args:
            data_path: Root directory containing metric folders
            file_pattern: Pattern to match CSV files
            
        Returns:
            Dictionary of {metric_name: {experiment_name: dataframe}}
        """
        print(f"🔍 Loading all datasets from {data_path}...")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        all_datasets = {}
        total_files = 0
        
        # Iterate through metric directories
        for metric_dir in data_path.iterdir():
            if not metric_dir.is_dir():
                continue
                
            metric_name = metric_dir.name
            print(f"  📊 Loading {metric_name}...")
            experiments = {}
            
            # Load all CSV files in the metric directory
            csv_files = list(metric_dir.glob(file_pattern))
            if not csv_files:
                print(f"    ⚠ No CSV files found in {metric_dir}")
                continue
            
            for csv_file in csv_files:
                exp_name = csv_file.stem  # filename without extension
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Basic validation
                    if len(df) == 0:
                        print(f"    ⚠ Empty file: {csv_file}")
                        continue
                    
                    # Ensure Minutes column exists
                    if 'Minutes' not in df.columns and 'Time' in df.columns:
                        # Try to create Minutes from Time if possible
                        df['Minutes'] = range(len(df))
                        print(f"    ℹ Created Minutes column for {exp_name}")
                    elif 'Minutes' not in df.columns:
                        df['Minutes'] = range(len(df))
                        print(f"    ℹ Created Minutes column for {exp_name}")
                    
                    experiments[exp_name] = df
                    total_files += 1
                    print(f"    ✅ {exp_name}: {len(df)} data points, {len(df.columns)} columns")
                    
                except Exception as e:
                    print(f"    ❌ Error loading {csv_file}: {e}")
                    continue
            
            if experiments:
                all_datasets[metric_name] = experiments
            else:
                print(f"    ⚠ No valid experiments found for {metric_name}")
        
        print(f"✅ Loaded {len(all_datasets)} metrics, {total_files} total files")
        return all_datasets
    
    def process_all_datasets(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Apply TUNA processing to all datasets
        
        Args:
            all_datasets: Dictionary of {metric_name: {experiment_name: dataframe}}
            
        Returns:
            TUNA results dictionary
        """
        print("\n" + "="*80)
        print("APPLYING TUNA PROCESSING TO ALL DATASETS")
        print("="*80)
        
        # Validate datasets first
        is_valid, errors = validate_datasets(all_datasets)
        if not is_valid:
            print("❌ Dataset validation failed:")
            for error in errors:
                print(f"  - {error}")
            raise ValueError("Dataset validation failed")
        
        # Train ML models for all metrics
        print("\n🤖 Training ML models...")
        self.trained_models = train_models_for_datasets(
            all_datasets, 
            random_state=self.random_state, 
            n_jobs=self.n_jobs
        )
        
        # Apply TUNA cleaning to all datasets
        print("\n🔧 Applying TUNA cleaning...")
        tuna_results = {}
        
        total_series = sum(len(experiments) for experiments in all_datasets.values())
        processed_series = 0
        
        for metric_name, experiments in all_datasets.items():
            print(f"\n📈 Processing {metric_name}...")
            
            # Get trained model and scaler for this metric
            model, scaler = self.trained_models.get(metric_name, (None, None))
            
            # Auto-detect metric type
            first_df = list(experiments.values())[0]
            metric_type = detect_metric_type(metric_name, first_df)
            print(f"  🔍 Detected metric type: {metric_type}")
            
            # Apply cleaning to each experimental condition
            metric_results = {}
            for exp_name, df in experiments.items():
                print(f"    🔄 Processing {exp_name}...")
                
                values = extract_values_by_type(df, metric_name, metric_type)
                
                cleaned_series, outlier_mask, stats = apply_tuna_to_single_series(
                    values, exp_name, model, scaler, 
                    penalty_factor=self.penalty_factor,
                    metric_name=metric_name,
                    metric_type=metric_type
                )
                
                metric_results[exp_name] = {
                    'original': values,
                    'cleaned': cleaned_series,
                    'outliers': outlier_mask,
                    'stats': stats,
                    'metric_type': metric_type,
                    'original_df': df.copy()  # Keep original dataframe
                }
                
                processed_series += 1
                print(f"      ✅ Outliers: {stats['outliers']}, Noise reduction: {stats['noise_reduction']:.1f}%")
            
            tuna_results[metric_name] = metric_results
            print(f"  ✅ Completed {metric_name} ({len(metric_results)} experiments)")
        
        self.tuna_results = tuna_results
        
        # Print summary
        print(f"\n📊 TUNA Processing Summary:")
        print_tuna_summary(tuna_results)
        
        return tuna_results
    
    def create_cleaned_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create new datasets with TUNA-cleaned values
        
        Returns:
            Dictionary of cleaned datasets in same format as input
        """
        print("\n" + "="*50)
        print("CREATING CLEANED DATASETS")
        print("="*50)
        
        cleaned_datasets = {}
        
        for metric_name, metric_results in self.tuna_results.items():
            print(f"📈 Creating cleaned dataset for {metric_name}...")
            
            cleaned_experiments = {}
            
            for exp_name, results in metric_results.items():
                # Get original dataframe structure
                original_df = results['original_df'].copy()
                cleaned_values = results['cleaned']
                
                # Create new dataframe with cleaned values
                cleaned_df = original_df.copy()
                
                # Replace metric values with cleaned ones
                exclude_cols = ['Time', 'Minutes', 'source']
                value_cols = [col for col in cleaned_df.columns if col not in exclude_cols]
                
                if len(value_cols) == 1:
                    # Single value column
                    cleaned_df[value_cols[0]] = cleaned_values[:len(cleaned_df)]
                else:
                    # Multiple value columns - distribute cleaned values proportionally
                    original_total = original_df[value_cols].sum(axis=1)
                    for col in value_cols:
                        # Maintain original proportions
                        proportion = original_df[col] / original_total
                        proportion = proportion.fillna(0)  # Handle division by zero
                        cleaned_df[col] = cleaned_values[:len(cleaned_df)] * proportion
                
                # Add metadata columns
                cleaned_df['tuna_processed'] = True
                cleaned_df['outliers_detected'] = results['outliers'][:len(cleaned_df)]
                
                cleaned_experiments[exp_name] = cleaned_df
                print(f"  ✅ {exp_name}: {len(cleaned_df)} points processed")
            
            cleaned_datasets[metric_name] = cleaned_experiments
        
        self.processed_datasets = cleaned_datasets
        print(f"✅ Created {len(cleaned_datasets)} cleaned datasets")
        return cleaned_datasets
    
    def export_cleaned_datasets(self, output_path: str, include_metadata: bool = True) -> str:
        """
        Export cleaned datasets to CSV files
        
        Args:
            output_path: Directory to save cleaned CSV files
            include_metadata: Whether to include TUNA metadata columns
            
        Returns:
            Path to exported data
        """
        print(f"\n💾 Exporting cleaned datasets to {output_path}...")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary file
        summary_data = []
        
        for metric_name, experiments in self.processed_datasets.items():
            metric_dir = output_path / metric_name
            metric_dir.mkdir(exist_ok=True)
            
            print(f"  📁 Exporting {metric_name}...")
            
            for exp_name, df in experiments.items():
                # Prepare dataframe for export
                export_df = df.copy()
                
                if not include_metadata:
                    # Remove TUNA metadata columns
                    metadata_cols = ['tuna_processed', 'outliers_detected']
                    export_df = export_df.drop(columns=[col for col in metadata_cols if col in export_df.columns])
                
                # Export to CSV
                csv_path = metric_dir / f"{exp_name}.csv"
                export_df.to_csv(csv_path, index=False)
                
                # Collect summary information
                tuna_stats = self.tuna_results[metric_name][exp_name]['stats']
                summary_data.append({
                    'metric': metric_name,
                    'experiment': exp_name,
                    'original_points': len(df),
                    'outliers_removed': tuna_stats['outliers'],
                    'outlier_percentage': tuna_stats['outlier_percentage'],
                    'noise_reduction': tuna_stats['noise_reduction'],
                    'correlation': tuna_stats['correlation'],
                    'ml_applied': tuna_stats.get('ml_applied', False),
                    'file_path': str(csv_path.relative_to(output_path))
                })
                
                print(f"    ✅ {exp_name}.csv ({len(export_df)} points)")
        
        # Export summary file
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_path / "tuna_processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Export configuration
        config_data = {
            'processing_parameters': {
                'penalty_factor': self.penalty_factor,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs
            },
            'metrics_processed': list(self.processed_datasets.keys()),
            'total_files_exported': len(summary_data),
            'ml_models_trained': len([m for m, s in self.trained_models.values() if m is not None])
        }
        
        config_path = output_path / "tuna_config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Exported {len(summary_data)} files to {output_path}")
        print(f"📋 Summary saved to: {summary_path}")
        print(f"⚙️ Configuration saved to: {config_path}")
        
        return str(output_path)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results"""
        if not self.tuna_results:
            return {"status": "not_processed"}
        
        total_series = sum(len(metric_results) for metric_results in self.tuna_results.values())
        total_outliers = sum(
            results['stats']['outliers'] 
            for metric_results in self.tuna_results.values()
            for results in metric_results.values()
        )
        
        ml_enhanced = sum(
            1 for metric_results in self.tuna_results.values()
            for results in metric_results.values()
            if results['stats'].get('ml_applied', False)
        )
        
        return {
            'status': 'processed',
            'total_metrics': len(self.tuna_results),
            'total_series': total_series,
            'total_outliers_removed': total_outliers,
            'ml_enhanced_series': ml_enhanced,
            'ml_enhancement_rate': (ml_enhanced / total_series * 100) if total_series > 0 else 0,
            'models_trained': len([m for m, s in self.trained_models.values() if m is not None])
        }


def process_datasets_for_failure_detection(data_path: str, output_path: str, 
                                         penalty_factor: float = 0.75,
                                         random_state: int = 42) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to process all datasets through TUNA pipeline
    
    Args:
        data_path: Input data directory
        output_path: Output directory for cleaned datasets
        penalty_factor: TUNA penalty factor
        random_state: Random state for ML models
        
    Returns:
        Tuple of (output_path, processing_summary)
    """
    processor = TunaDataProcessor(
        penalty_factor=penalty_factor,
        random_state=random_state
    )
    
    # Load all datasets
    all_datasets = processor.load_all_datasets(data_path)
    
    # Process through TUNA
    processor.process_all_datasets(all_datasets)
    
    # Create cleaned datasets
    processor.create_cleaned_datasets()
    
    # Export cleaned datasets
    exported_path = processor.export_cleaned_datasets(output_path)
    
    # Get summary
    summary = processor.get_processing_summary()
    
    return exported_path, summary