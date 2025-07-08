"""
Data processing utilities for TUNA analysis system
Handles data extraction, normalization, and preprocessing for different metric types
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


def extract_memory_values(df: pd.DataFrame) -> np.ndarray:
    """Extract memory values from datasets (handles both single and multi-column data)"""
    exclude_cols = ['Time', 'Minutes', 'source']
    value_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(value_cols) == 1:
        return df[value_cols[0]].values
    else:
        return df[value_cols].sum(axis=1).values


def extract_tcp_values(df: pd.DataFrame, metric_name: str) -> np.ndarray:
    """Extract TCP values from datasets (handles both single and multi-column data)"""
    exclude_cols = ['Time', 'Minutes', 'source']
    value_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(value_cols) == 1:
        # Single column (like individual service retransmission data)
        return df[value_cols[0]].values
    else:
        # Multiple columns - for SRTT data, create cumulative metrics
        if 'srtt' in metric_name.lower():
            # For SRTT data, group by service and sum all latency buckets
            service_groups = {}
            for col in value_cols:
                if '_client_' in col or '_server_' in col:
                    service_name = col.split('_client_')[0] if '_client_' in col else col.split('_server_')[0]
                else:
                    service_name = col
                if service_name not in service_groups:
                    service_groups[service_name] = []
                service_groups[service_name].append(col)
            
            # Sum the first service group (or all if you want total)
            if service_groups:
                first_service = list(service_groups.keys())[0]
                return df[service_groups[first_service]].fillna(0).sum(axis=1).values
            else:
                return df[value_cols].fillna(0).sum(axis=1).values
        else:
            # For other multi-column data, sum all columns
            return df[value_cols].fillna(0).sum(axis=1).values


def extract_disk_values(df: pd.DataFrame, metric_name: str) -> np.ndarray:
    """Extract disk I/O values from datasets"""
    exclude_cols = ['Time', 'Minutes', 'source']
    value_cols = [col for col in df.columns if col not in exclude_cols]
    
    if metric_name == 'BlockLatency':
        return df[value_cols[0]].values
    else:
        return df[value_cols].sum(axis=1).values


def extract_cpu_values(df: pd.DataFrame) -> np.ndarray:
    """Calculate total CPU from all CPU columns"""
    exclude_cols = ['Time', 'Minutes', 'source']
    cpu_cols = [col for col in df.columns if col not in exclude_cols]
    return df[cpu_cols].sum(axis=1).values


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a series to 0-1 range"""
    if len(series) == 0:
        return series
    
    min_val = series.min()
    max_val = series.max()
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    else:
        return pd.Series(np.zeros(len(series)), index=series.index if hasattr(series, 'index') else None)


def add_phase_column(df: pd.DataFrame, delay_minutes: int = 30, duration_minutes: int = 50) -> pd.DataFrame:
    """Add phase column based on experiment timeline"""
    df = df.copy()
    df['phase'] = 'before'
    df.loc[(df['Minutes'] >= delay_minutes) & 
           (df['Minutes'] <= delay_minutes + duration_minutes), 'phase'] = 'during'
    df.loc[df['Minutes'] > delay_minutes + duration_minutes, 'phase'] = 'after'
    return df


def get_metric_label(metric_name: str, metric_type: str = 'general') -> str:
    """Get appropriate y-axis label based on metric name and type"""
    
    # Memory labels
    memory_labels = {
        'MemAvailable': 'Memory Available (KB)',
        'MemCache': 'Memory Cache (KB)', 
        'MemUtil': 'Memory Utilization (%)',
        'Memory Utilization': 'Memory Utilization (%)',
        'Memory Cache': 'Memory Cache (KB)',
        'Memory Available': 'Memory Available (KB)',
        'Memory Usage': 'Memory Usage (KB)',
        'Memory Buffer': 'Memory Buffer (KB)',
        'Memory Free': 'Memory Free (KB)',
    }
    
    # CPU labels
    cpu_labels = {
        'IOWait': 'CPU IOWait (msec)',
        'IRQ': 'CPU IRQ (msec)',
        'System': 'CPU System (msec)', 
        'User': 'CPU User (msec)',
        'Utilization': 'CPU Utilization (%)'
    }
    
    # Disk labels
    disk_labels = {
        'BlockLatency': 'Block Latency (ms)',
        'ReadBytes': 'Read Bytes (KB)',
        'WriteBytes': 'Write Bytes (KB)',
        'DiskUtil': 'Disk Utilization (%)',
        'IOPS': 'I/O Operations per Second',
        'ThroughputRead': 'Read Throughput (MB/s)',
        'ThroughputWrite': 'Write Throughput (MB/s)'
    }
    
    # TCP labels - determined dynamically
    if 'srtt' in metric_name.lower():
        tcp_label = 'SRTT Values'
    elif any(service in metric_name.lower() for service in ['apigateway', 'customers', 'visits', 'vets']):
        tcp_label = 'Retransmission Packets'
    else:
        tcp_label = 'TCP Values'
    
    # Select appropriate label dictionary based on metric type
    if metric_type == 'memory':
        return memory_labels.get(metric_name, 'Memory Value')
    elif metric_type == 'cpu':
        return cpu_labels.get(metric_name, 'Value')
    elif metric_type == 'disk':
        return disk_labels.get(metric_name, 'Disk I/O Value')
    elif metric_type == 'tcp':
        return tcp_label
    else:
        # Try all dictionaries for backward compatibility
        for labels in [memory_labels, cpu_labels, disk_labels]:
            if metric_name in labels:
                return labels[metric_name]
        return 'Value'


def detect_metric_type(metric_name: str, df: pd.DataFrame) -> str:
    """Auto-detect metric type based on metric name and data characteristics"""
    metric_lower = metric_name.lower()
    
    # Check column names for hints
    columns = df.columns.tolist()
    
    if any(mem_term in metric_lower for mem_term in ['mem', 'memory']):
        return 'memory'
    elif any(cpu_term in metric_lower for cpu_term in ['cpu', 'iowait', 'irq', 'system', 'user', 'utilization']):
        return 'cpu'
    elif any(disk_term in metric_lower for disk_term in ['disk', 'block', 'read', 'write', 'iops', 'throughput']):
        return 'disk'
    elif any(tcp_term in metric_lower for tcp_term in ['tcp', 'srtt', 'retrans']) or \
         any(service in metric_lower for service in ['apigateway', 'customers', 'visits', 'vets']):
        return 'tcp'
    else:
        return 'general'


def prepare_combined_dataframe(experiment_values: np.ndarray, baseline_values: np.ndarray,
                             experiment_df: pd.DataFrame, baseline_df: pd.DataFrame,
                             delay_minutes: int = 30, duration_minutes: int = 50) -> pd.DataFrame:
    """Prepare combined normalized dataframe for analysis"""
    
    # Create normalized series
    experiment_norm = normalize_series(pd.Series(experiment_values))
    baseline_norm = normalize_series(pd.Series(baseline_values))
    
    # Ensure consistent length
    min_length = min(len(experiment_norm), len(baseline_norm), 
                    len(experiment_df['Minutes']), len(baseline_df['Minutes']))
    
    # Create combined dataframe
    df_combined = pd.DataFrame({
        'Baseline': baseline_norm.iloc[:min_length],
        'Experiment': experiment_norm.iloc[:min_length],
        'Minutes': experiment_df['Minutes'].iloc[:min_length],
        'Baseline_Minutes': baseline_df['Minutes'].iloc[:min_length]
    })
    
    # Calculate difference
    df_combined['difference'] = df_combined['Experiment'] - df_combined['Baseline']
    
    # Add phase information
    df_combined = add_phase_column(df_combined, delay_minutes, duration_minutes)
    
    return df_combined


def extract_values_by_type(df: pd.DataFrame, metric_name: str, metric_type: str = None) -> np.ndarray:
    """Extract values based on detected or specified metric type"""
    
    if metric_type is None:
        metric_type = detect_metric_type(metric_name, df)
    
    if metric_type == 'memory':
        return extract_memory_values(df)
    elif metric_type == 'cpu':
        return extract_cpu_values(df)
    elif metric_type == 'disk':
        return extract_disk_values(df, metric_name)
    elif metric_type == 'tcp':
        return extract_tcp_values(df, metric_name)
    else:
        # Default fallback
        return extract_memory_values(df)


def validate_datasets(all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[bool, List[str]]:
    """Validate that all datasets have required structure"""
    errors = []
    
    for metric_name, experiments in all_datasets.items():
        if 'baseline' not in experiments:
            errors.append(f"Missing baseline for metric: {metric_name}")
        
        for exp_name, df in experiments.items():
            if 'Minutes' not in df.columns:
                errors.append(f"Missing 'Minutes' column in {metric_name}.{exp_name}")
            
            if len(df) == 0:
                errors.append(f"Empty dataset: {metric_name}.{exp_name}")
    
    return len(errors) == 0, errors


def get_experiment_summary(all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
    """Get summary statistics about the datasets"""
    summary = {
        'total_metrics': len(all_datasets),
        'metrics': list(all_datasets.keys()),
        'experiments': set(),
        'data_points': {},
        'metric_types': {}
    }
    
    for metric_name, experiments in all_datasets.items():
        summary['metric_types'][metric_name] = detect_metric_type(metric_name, list(experiments.values())[0])
        summary['data_points'][metric_name] = {}
        
        for exp_name, df in experiments.items():
            summary['experiments'].add(exp_name)
            summary['data_points'][metric_name][exp_name] = len(df)
    
    summary['experiments'] = list(summary['experiments'])
    
    return summary