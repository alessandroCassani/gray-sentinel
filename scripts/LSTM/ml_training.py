"""
Machine Learning model training module for TUNA system
Handles RandomForest training for time series prediction and enhancement
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from data_utils import extract_values_by_type, detect_metric_type
from tuna_core import detect_outliers_tuna, create_features_for_ml
import warnings
warnings.filterwarnings('ignore')


class TunaMLTrainer:
    """Machine Learning trainer for TUNA system"""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trained_models = {}
        self.scalers = {}
    
    def train_model_for_metric(self, metric_name: str, all_experiments: Dict[str, pd.DataFrame], 
                              metric_type: Optional[str] = None) -> Tuple[Optional[RandomForestRegressor], Optional[StandardScaler]]:
        """
        Train RandomForest on stable periods across all experiments for a specific metric
        
        Args:
            metric_name: Name of the metric to train on
            all_experiments: Dictionary of experiment dataframes
            metric_type: Type of metric (auto-detected if None)
        
        Returns:
            Tuple of (trained_model, fitted_scaler) or (None, None) if training failed
        """
        print(f"Training model for {metric_name}...")
        
        if metric_type is None:
            # Auto-detect metric type from first experiment
            first_df = list(all_experiments.values())[0]
            metric_type = detect_metric_type(metric_name, first_df)
        
        X_stable_all = []
        y_stable_all = []
        
        # Aggregate stable training data from all experimental conditions
        for exp_name, df in all_experiments.items():
            try:
                values = extract_values_by_type(df, metric_name, metric_type)
                outlier_mask = detect_outliers_tuna(values)
                stable_mask = ~outlier_mask
                
                features = create_features_for_ml(values, exp_name)
                # Align feature window with stability detection
                stable_features_mask = stable_mask[10:]  # Account for 10-point window
                stable_features = features[stable_features_mask]
                stable_targets = values[10:][stable_features_mask]
                
                if len(stable_features) > 0:
                    # Apply local smoothing to targets for better generalization
                    smoothed_targets = self._apply_local_smoothing(stable_targets)
                    
                    X_stable_all.extend(stable_features)
                    y_stable_all.extend(smoothed_targets)
                    
                print(f"  Added {len(stable_features)} stable samples from {exp_name}")
                
            except Exception as e:
                print(f"  ⚠ Failed to extract features from {exp_name}: {str(e)}")
                continue
        
        # Ensure sufficient training data
        if len(X_stable_all) < 10:
            print(f"  ⚠ Insufficient training data for {metric_name} ({len(X_stable_all)} samples)")
            return None, None
        
        X_stable_all = np.array(X_stable_all)
        y_stable_all = np.array(y_stable_all)
        
        print(f"  Training on {len(X_stable_all)} stable samples across {len(all_experiments)} experiments")
        
        try:
            # Standardize features for optimal model performance
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_stable_all)
            
            # Train model with hyperparameter optimization
            model = self._train_random_forest(X_scaled, y_stable_all)
            
            # Store trained assets
            self.trained_models[metric_name] = model
            self.scalers[metric_name] = scaler
            
            # Evaluate model performance
            train_score = model.score(X_scaled, y_stable_all)
            print(f"  ✓ Model trained successfully. R² score: {train_score:.3f}")
            
            return model, scaler
            
        except Exception as e:
            print(f"  ✗ Model training failed for {metric_name}: {str(e)}")
            return None, None
    
    def _apply_local_smoothing(self, targets: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply local median smoothing to target values"""
        smoothed_targets = []
        for j in range(len(targets)):
            start_idx = max(0, j - window_size // 2)
            end_idx = min(len(targets), j + window_size // 2 + 1)
            local_values = targets[start_idx:end_idx]
            smoothed_targets.append(np.median(local_values))
        return np.array(smoothed_targets)
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """Train RandomForest with hyperparameter optimization"""
        
        # Hyperparameter grid for model optimization
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use smaller grid for small datasets
        if len(X) < 100:
            param_grid = {
                'n_estimators': [100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        
        base_model = RandomForestRegressor(
            criterion='squared_error',
            bootstrap=True,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        # Grid search with cross-validation for robust parameter selection
        cv_folds = min(5, max(3, len(X) // 20))  # Adaptive CV folds
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=cv_folds,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def train_all_metrics(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Tuple[Optional[RandomForestRegressor], Optional[StandardScaler]]]:
        """
        Train models for all metrics in the dataset
        
        Args:
            all_datasets: Dictionary of {metric_name: {experiment_name: dataframe}}
        
        Returns:
            Dictionary of {metric_name: (model, scaler)}
        """
        print("Training ML models for all metrics...")
        print("=" * 50)
        
        trained_assets = {}
        
        for metric_name, experiments in all_datasets.items():
            model, scaler = self.train_model_for_metric(metric_name, experiments)
            trained_assets[metric_name] = (model, scaler)
        
        # Print training summary
        successful_models = sum(1 for model, scaler in trained_assets.values() if model is not None)
        print(f"\n✓ Training completed: {successful_models}/{len(all_datasets)} models trained successfully")
        
        return trained_assets
    
    def get_model_info(self, metric_name: str) -> Dict[str, Any]:
        """Get information about a trained model"""
        if metric_name not in self.trained_models:
            return {"status": "not_trained"}
        
        model = self.trained_models[metric_name]
        scaler = self.scalers[metric_name]
        
        return {
            "status": "trained",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features,
            "feature_count": scaler.n_features_in_ if scaler else None,
            "oob_score": getattr(model, 'oob_score_', None)
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        total_models = len(self.trained_models)
        
        if total_models == 0:
            return {"total_models": 0, "trained_metrics": []}
        
        model_info = {}
        for metric_name in self.trained_models:
            model_info[metric_name] = self.get_model_info(metric_name)
        
        return {
            "total_models": total_models,
            "trained_metrics": list(self.trained_models.keys()),
            "model_details": model_info
        }


def create_tuna_trainer(random_state: int = 42, n_jobs: int = -1) -> TunaMLTrainer:
    """Factory function to create a TUNA ML trainer"""
    return TunaMLTrainer(random_state=random_state, n_jobs=n_jobs)


def train_models_for_datasets(all_datasets: Dict[str, Dict[str, pd.DataFrame]], 
                             random_state: int = 42, n_jobs: int = -1) -> Dict[str, Tuple[Optional[RandomForestRegressor], Optional[StandardScaler]]]:
    """
    Convenience function to train models for all datasets
    
    Args:
        all_datasets: Dictionary of {metric_name: {experiment_name: dataframe}}
        random_state: Random state for reproducibility
        n_jobs: Number of parallel jobs
    
    Returns:
        Dictionary of {metric_name: (model, scaler)}
    """
    trainer = create_tuna_trainer(random_state=random_state, n_jobs=n_jobs)
    return trainer.train_all_metrics(all_datasets)