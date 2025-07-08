"""
TUNA (Time-series Unsupervised Noise Analysis) core module
Handles outlier detection and data cleaning for time series analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def detect_outliers_tuna(timeseries: np.ndarray, window_size: int = 3, threshold: float = 0.85, 
                        min_absolute_range: Optional[float] = None) -> np.ndarray:
    """
    TUNA's relative range outlier detection with stability improvements
    Formula: (max - min) / mean > threshold
    Mark all values in unstable windows as outliers
    
    Args:
        timeseries: Time series data
        window_size: Size of sliding window for detection
        threshold: Relative range threshold for outlier detection
        min_absolute_range: Minimum absolute range to avoid hypersensitivity
    
    Returns:
        Boolean array marking outliers
    """
    outlier_mask = np.zeros(len(timeseries), dtype=bool)
    
    # Calculate adaptive minimum absolute range if not provided
    if min_absolute_range is None:
        # Use 10% of the overall time series standard deviation as minimum range
        min_absolute_range = 0.1 * np.std(timeseries)
    
    # Slide window across time series to detect unstable periods
    for i in range(len(timeseries) - window_size + 1):
        window = timeseries[i:i + window_size]
        window_mean = np.mean(window)
        window_range = np.max(window) - np.min(window)
        
        if window_mean > 0:
            # Apply TUNA relative range criterion
            relative_range = window_range / window_mean
            
            # Additional filter: require minimum absolute range to avoid hypersensitivity
            if relative_range > threshold and window_range > min_absolute_range:
                # Mark all values within unstable window as outliers
                for j in range(window_size):
                    actual_idx = i + j
                    outlier_mask[actual_idx] = True
    
    return outlier_mask


def create_features_for_ml(timeseries: np.ndarray, experiment_type: str, window_size: int = 10) -> np.ndarray:
    """Create features for RandomForest training"""
    features = []
    
    # Extract temporal and statistical features from sliding windows
    for i in range(window_size, len(timeseries)):
        window = timeseries[i-window_size:i]
        
        # Statistical features from recent history
        feature_vector = [
            np.mean(window),                    # Rolling average
            np.std(window),                     # Variability measure
            np.median(window),                  # Robust central tendency
            np.max(window) - np.min(window),    # Range indicator
            timeseries[i-1],                    # Previous value dependency
            i / len(timeseries),                # Temporal position
        ]
        
        # Experiment context encoding for cross-experiment learning
        exp_features = [0, 0, 0, 0, 0]
        if experiment_type == "baseline":
            exp_features[0] = 1
        elif experiment_type == "cpu_stress":
            exp_features[1] = 1
        elif experiment_type == "delay":
            exp_features[2] = 1
        elif experiment_type == "mem_stress":
            exp_features[3] = 1
        elif experiment_type == "net_loss":
            exp_features[4] = 1
        
        feature_vector.extend(exp_features)
        features.append(feature_vector)
    
    return np.array(features)


def apply_penalty(timeseries: np.ndarray, outlier_mask: np.ndarray, penalty_factor: float = 0.75) -> np.ndarray:
    """Apply penalty to outliers by reducing their magnitude"""
    cleaned_series = timeseries.copy()
    
    # Reduce outlier magnitude by penalty factor
    for i in range(len(timeseries)):
        if outlier_mask[i]:
            original_value = timeseries[i]
            # Simple penalty: reduce magnitude by factor
            cleaned_series[i] = original_value * penalty_factor
    
    return cleaned_series


def apply_tuna_to_single_series(values: np.ndarray, exp_name: str, model: Optional[RandomForestRegressor], 
                               scaler: Optional[StandardScaler], penalty_factor: float = 0.75, 
                               metric_name: str = "", metric_type: str = "general") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply TUNA cleaning to a single time series with ML + penalty
    
    Args:
        values: Time series values
        exp_name: Experiment name
        model: Trained RandomForest model (optional)
        scaler: Fitted StandardScaler (optional)
        penalty_factor: Factor to reduce outlier magnitude
        metric_name: Name of the metric for specific handling
        metric_type: Type of metric (memory, cpu, disk, tcp, general)
    
    Returns:
        Tuple of (cleaned_series, outlier_mask, cleaning_stats)
    """
    
    # Check data characteristics for smart ML application
    unique_values = len(np.unique(values))
    data_range = np.max(values) - np.min(values)
    zero_percentage = np.sum(values == 0) / len(values) * 100
    
    # Enhanced discrete data detection for different metric types
    is_discrete_data = False
    
    if metric_type == 'tcp':
        # For TCP retransmission data
        is_discrete_data = (
            unique_values <= 10 and data_range <= 50  # Low variety and range
            or zero_percentage > 60  # High percentage of zeros (common in retrans data)
            or (unique_values <= 20 and all(x == int(x) for x in np.unique(values) if not np.isnan(x)))  # All integers
        )
    elif metric_type in ['cpu', 'memory']:
        # For CPU/Memory metrics, less likely to be discrete
        is_discrete_data = unique_values <= 5 and zero_percentage > 80
    else:
        # General case
        is_discrete_data = unique_values <= 10 and zero_percentage > 50
    
    # Phase 1: Identify unstable measurements with adjusted parameters
    if metric_type == 'tcp' and (any(service in metric_name.lower() for service in ['apigateway', 'customers', 'vets', 'visits']) or zero_percentage > 50):
        # For retransmission data, use more sensitive outlier detection
        outlier_mask = detect_outliers_tuna(values, window_size=5, threshold=0.65)
    else:
        # Standard outlier detection for continuous metrics
        outlier_mask = detect_outliers_tuna(values)
    
    stable_mask = ~outlier_mask
    outliers_count = np.sum(outlier_mask)
    
    # Phase 2: Apply penalty to outliers FIRST
    cleaned_series = apply_penalty(values, outlier_mask, penalty_factor)
    
    # Phase 3: Apply ML enhancement only for appropriate data types
    if not is_discrete_data and model is not None and scaler is not None:
        try:
            # Create features for ML prediction
            features = create_features_for_ml(values, exp_name)
            features_scaled = scaler.transform(features)
            ml_predictions = model.predict(features_scaled)
            
            # Apply ML predictions for stable periods only
            for i, prediction in enumerate(ml_predictions):
                actual_idx = i + 10  # Account for feature window offset
                if actual_idx < len(cleaned_series) and stable_mask[actual_idx]:
                    cleaned_series[actual_idx] = prediction
            
            # Ensure non-negative values for count data
            if metric_type == 'tcp' and any(service in metric_name.lower() for service in ['apigateway', 'customers', 'vets', 'visits']):
                cleaned_series = np.maximum(0, cleaned_series)
            
            print(f"  ✓ Applied ML + penalty (factor: {penalty_factor}) for {metric_name} - {exp_name}")
            
        except Exception as e:
            print(f"  ⚠ ML enhancement failed for {metric_name} - {exp_name}: {str(e)}")
    else:
        if is_discrete_data:
            print(f"  ℹ Applied penalty-only (factor: {penalty_factor}) for {metric_name} - {exp_name} (discrete data)")
        else:
            print(f"  ℹ Applied penalty-only (factor: {penalty_factor}) for {metric_name} - {exp_name} (no model)")
    
    # Calculate cleaning statistics
    original_std = np.std(values)
    cleaned_std = np.std(cleaned_series)
    noise_reduction = (original_std - cleaned_std) / original_std * 100 if original_std > 0 else 0
    correlation = np.corrcoef(values, cleaned_series)[0, 1] if len(values) > 1 else 1.0
    
    cleaning_stats = {
        'outliers': outliers_count,
        'outlier_percentage': (outliers_count / len(values)) * 100,
        'noise_reduction': noise_reduction,
        'correlation': correlation,
        'mean_before': np.mean(values),
        'mean_after': np.mean(cleaned_series),
        'std_before': original_std,
        'std_after': cleaned_std,
        'zero_percentage': zero_percentage,
        'unique_values': unique_values,
        'data_type': 'discrete' if is_discrete_data else 'continuous',
        'penalty_factor': penalty_factor,
        'ml_applied': not is_discrete_data and model is not None
    }
    
    return cleaned_series, outlier_mask, cleaning_stats


def calculate_cleaning_effectiveness(stats: Dict[str, Any]) -> float:
    """Calculate overall cleaning effectiveness score"""
    noise_score = min(stats['noise_reduction'] / 50.0, 1.0)  # Normalize to 0-1, cap at 50%
    correlation_score = stats['correlation']
    outlier_score = min(stats['outlier_percentage'] / 20.0, 1.0)  # Normalize to 0-1, cap at 20%
    
    # Weighted combination
    effectiveness = (0.4 * noise_score + 0.4 * correlation_score + 0.2 * outlier_score)
    return effectiveness


def print_tuna_summary(tuna_results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """Print summary table of cleaning effectiveness"""
    print("\nTUNA Results Summary:")
    print("=" * 80)
    print(f"{'Metric':<20} {'Experiment':<15} {'Outliers':<10} {'Noise Red%':<12} {'Correlation':<12} {'ML Used':<8}")
    print("-" * 80)
    
    # Tabulate results across all metrics and experiments
    for metric_name, metric_results in tuna_results.items():
        for exp_name, results in metric_results.items():
            stats = results['stats']
            ml_used = "Yes" if stats.get('ml_applied', False) else "No"
            print(f"{metric_name:<20} {exp_name:<15} {stats['outliers']:<10} "
                  f"{stats['noise_reduction']:>10.1f}% {stats['correlation']:>11.3f} {ml_used:<8}")


def get_tuna_summary_stats(tuna_results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Get aggregate statistics from TUNA results"""
    total_outliers = 0
    total_points = 0
    noise_reductions = []
    correlations = []
    ml_usage = 0
    total_series = 0
    
    for metric_name, metric_results in tuna_results.items():
        for exp_name, results in metric_results.items():
            stats = results['stats']
            total_outliers += stats['outliers']
            total_points += len(results['original'])
            noise_reductions.append(stats['noise_reduction'])
            correlations.append(stats['correlation'])
            if stats.get('ml_applied', False):
                ml_usage += 1
            total_series += 1
    
    return {
        'total_outliers': total_outliers,
        'total_points': total_points,
        'outlier_percentage': (total_outliers / total_points * 100) if total_points > 0 else 0,
        'avg_noise_reduction': np.mean(noise_reductions) if noise_reductions else 0,
        'avg_correlation': np.mean(correlations) if correlations else 0,
        'ml_usage_percentage': (ml_usage / total_series * 100) if total_series > 0 else 0,
        'total_series': total_series
    }