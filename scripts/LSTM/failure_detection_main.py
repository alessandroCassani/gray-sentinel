#!/usr/bin/env python3
"""
Complete TUNA + Failure Detection Pipeline
1. Load all datasets
2. Apply TUNA cleaning to each metric
3. Export cleaned datasets
4. Train LSTM/GRU models for failure detection
5. Evaluate and compare models
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_processor import TunaDataProcessor, process_datasets_for_failure_detection
from failure_ml_models import train_failure_detection_models, TF_AVAILABLE


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='TUNA + Failure Detection Pipeline')
    
    # Data paths
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to raw datasets directory')
    parser.add_argument('--output-path', type=str, default='tuna_failure_output',
                       help='Output directory for all results')
    
    # TUNA parameters
    parser.add_argument('--penalty-factor', type=float, default=0.75,
                       help='TUNA penalty factor for outliers (default: 0.75)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    # Failure detection parameters
    parser.add_argument('--sequence-length', type=int, default=30,
                       help='Sequence length for LSTM/GRU (default: 30)')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                       help='Prediction horizon for failure detection (default: 5)')
    parser.add_argument('--models', nargs='+', default=['lstm', 'gru'],
                       choices=['lstm', 'gru', 'attention'],
                       help='Model types to train (default: lstm gru)')
    
    # Processing options
    parser.add_argument('--skip-tuna', action='store_true',
                       help='Skip TUNA processing (use existing cleaned data)')
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML training (only do TUNA processing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("🔬 TUNA + Failure Detection Pipeline")
    print("=" * 60)
    
    # Check TensorFlow availability for ML training
    if not args.skip_ml and not TF_AVAILABLE:
        print("❌ TensorFlow not available. Install with: pip install tensorflow>=2.8.0")
        print("   Or use --skip-ml to only run TUNA processing")
        return 1
    
    try:
        # Create output directories
        output_path = Path(args.output_path)
        cleaned_data_path = output_path / "cleaned_datasets"
        models_path = output_path / "failure_models"
        
        output_path.mkdir(exist_ok=True)
        cleaned_data_path.mkdir(exist_ok=True)
        models_path.mkdir(exist_ok=True)
        
        # Configuration summary
        config = {
            'data_path': args.data_path,
            'output_path': str(args.output_path),
            'tuna_parameters': {
                'penalty_factor': args.penalty_factor,
                'random_state': args.random_state
            },
            'failure_detection_parameters': {
                'sequence_length': args.sequence_length,
                'prediction_horizon': args.prediction_horizon,
                'model_types': args.models
            },
            'processing_options': {
                'skip_tuna': args.skip_tuna,
                'skip_ml': args.skip_ml
            }
        }
        
        print(f"⚙️  Configuration:")
        for section, params in config.items():
            if isinstance(params, dict):
                print(f"  {section}:")
                for key, value in params.items():
                    print(f"    - {key}: {value}")
            else:
                print(f"  {section}: {params}")
        
        # Save configuration
        config_path = output_path / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📄 Configuration saved to {config_path}")
        
        results = {}
        
        # PHASE 1: TUNA PROCESSING
        if not args.skip_tuna:
            print(f"\n{'='*60}")
            print("PHASE 1: TUNA DATA PROCESSING")
            print(f"{'='*60}")
            
            # Process datasets through TUNA
            exported_path, tuna_summary = process_datasets_for_failure_detection(
                data_path=args.data_path,
                output_path=str(cleaned_data_path),
                penalty_factor=args.penalty_factor,
                random_state=args.random_state
            )
            
            results['tuna_processing'] = {
                'exported_path': exported_path,
                'summary': tuna_summary
            }
            
            print(f"\n✅ TUNA processing completed!")
            print(f"📁 Cleaned datasets saved to: {cleaned_data_path}")
        
        else:
            print(f"\n⏭️  Skipping TUNA processing (using existing data from {cleaned_data_path})")
            if not cleaned_data_path.exists():
                print(f"❌ Cleaned data path does not exist: {cleaned_data_path}")
                return 1
        
        # PHASE 2: FAILURE DETECTION MODEL TRAINING
        if not args.skip_ml:
            print(f"\n{'='*60}")
            print("PHASE 2: FAILURE DETECTION MODEL TRAINING")
            print(f"{'='*60}")
            
            if not TF_AVAILABLE:
                print("❌ TensorFlow not available for ML training")
                return 1
            
            # Train failure detection models
            ml_results = train_failure_detection_models(
                cleaned_data_path=str(cleaned_data_path),
                output_path=str(models_path),
                sequence_length=args.sequence_length,
                model_types=args.models
            )
            
            results['failure_detection'] = ml_results
            
            print(f"\n✅ Failure detection training completed!")
            print(f"🤖 Models saved to: {models_path}")
            
            # Print final model comparison
            print(f"\n📊 Final Model Performance:")
            print(ml_results['comparison'].to_string(index=False))
            
        else:
            print(f"\n⏭️  Skipping ML training")
        
        # PHASE 3: COMPREHENSIVE SUMMARY
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETION SUMMARY")
        print(f"{'='*60}")
        
        if 'tuna_processing' in results:
            tuna_summary = results['tuna_processing']['summary']
            print(f"\n🔧 TUNA Processing Results:")
            print(f"  - Metrics processed: {tuna_summary['total_metrics']}")
            print(f"  - Time series processed: {tuna_summary['total_series']}")
            print(f"  - Total outliers removed: {tuna_summary['total_outliers_removed']}")
            print(f"  - ML enhancement rate: {tuna_summary['ml_enhancement_rate']:.1f}%")
        
        if 'failure_detection' in results:
            ml_summary = results['failure_detection']
            best_model = ml_summary['comparison'].loc[ml_summary['comparison']['F1-Score'].idxmax()]
            print(f"\n🤖 Failure Detection Results:")
            print(f"  - Models trained: {len(args.models)}")
            print(f"  - Dataset size: {ml_summary['data_shape']}")
            print(f"  - Best model: {best_model['Model']} (F1={best_model['F1-Score']:.4f})")
            print(f"  - Best model saved: {ml_summary['best_model_path']}")
        
        # Save complete results
        results_path = output_path / "complete_results.json"
        
        # Prepare serializable results
        serializable_results = {}
        for key, value in results.items():
            if key == 'failure_detection':
                # Remove non-serializable objects
                serializable_results[key] = {
                    'comparison': value['comparison'].to_dict(),
                    'best_model_path': value['best_model_path'],
                    'data_shape': value['data_shape']
                }
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n📄 Complete results saved to: {results_path}")
        print(f"📁 All outputs available in: {output_path}")
        
        print(f"\n✅ Pipeline completed successfully! 🎉")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_example_config() -> Dict[str, Any]:
    """Create example configuration for the pipeline"""
    return {
        "description": "TUNA + Failure Detection Pipeline Configuration",
        "data_processing": {
            "penalty_factor": 0.75,
            "random_state": 42,
            "include_metadata": True
        },
        "failure_detection": {
            "sequence_length": 30,
            "prediction_horizon": 5,
            "model_types": ["lstm", "gru", "attention"],
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "test_size": 0.2,
                "early_stopping_patience": 15
            },
            "model_architecture": {
                "hidden_units": 64,
                "dropout_rate": 0.3,
                "l2_regularization": 0.01
            }
        },
        "experiment_setup": {
            "stress_start_minute": 30,
            "stress_duration_minutes": 50,
            "baseline_experiment": "baseline",
            "failure_experiments": ["cpu_stress", "mem_stress", "delay", "net_loss"]
        }
    }


if __name__ == "__main__":
    # Create example config if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        example_config = create_example_config()
        with open("failure_detection_config.json", 'w') as f:
            json.dump(example_config, f, indent=2)
        print("📄 Example configuration created: failure_detection_config.json")
        sys.exit(0)
    
    sys.exit(main())