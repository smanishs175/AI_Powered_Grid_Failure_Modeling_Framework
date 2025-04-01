"""
Data quality metrics for the Data Management Module.

This module provides functions to assess the quality of different types 
of data used in the Grid Failure Modeling Framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_data_completeness(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate data completeness metrics.
    
    Args:
        data: DataFrame to evaluate
        
    Returns:
        Dict: Dictionary of completeness metrics per column
    """
    metrics = {}
    
    # Overall completeness
    metrics['overall_completeness'] = (1.0 - data.isna().mean().mean()) * 100
    
    # Per-column completeness
    column_metrics = {}
    for column in data.columns:
        completeness = (1.0 - data[column].isna().mean()) * 100
        column_metrics[column] = completeness
    
    metrics['column_completeness'] = column_metrics
    
    return metrics

def calculate_temporal_coverage(data: pd.DataFrame, 
                                date_column: str,
                                groupby_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate temporal coverage metrics.
    
    Args:
        data: DataFrame to evaluate
        date_column: Name of the column containing dates/timestamps
        groupby_column: Optional column to group by (e.g., station_id)
        
    Returns:
        Dict: Dictionary of temporal coverage metrics
    """
    metrics = {}
    
    # Ensure datetime format
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Overall date range
    min_date = data[date_column].min()
    max_date = data[date_column].max()
    date_range = max_date - min_date
    
    metrics['start_date'] = min_date.strftime('%Y-%m-%d')
    metrics['end_date'] = max_date.strftime('%Y-%m-%d')
    metrics['total_days'] = date_range.days
    
    # Calculate coverage if groupby is provided
    if groupby_column:
        groups = data.groupby(groupby_column)
        group_metrics = {}
        
        for group_name, group_data in groups:
            group_min = group_data[date_column].min()
            group_max = group_data[date_column].max()
            group_range = group_max - group_min
            
            # Calculate expected number of data points based on range
            # Assuming daily data
            expected_days = group_range.days + 1
            actual_days = len(group_data[date_column].dt.date.unique())
            
            coverage = (actual_days / expected_days) * 100 if expected_days > 0 else 0
            
            group_metrics[str(group_name)] = {
                'start_date': group_min.strftime('%Y-%m-%d'),
                'end_date': group_max.strftime('%Y-%m-%d'),
                'total_days': group_range.days,
                'expected_points': expected_days,
                'actual_points': actual_days,
                'coverage_percent': coverage
            }
        
        metrics['group_coverage'] = group_metrics
        
        # Average coverage across groups
        avg_coverage = np.mean([g['coverage_percent'] for g in group_metrics.values()])
        metrics['average_group_coverage'] = avg_coverage
    
    # Calculate overall coverage
    # Count unique days in the dataset
    unique_days = len(data[date_column].dt.date.unique())
    expected_days = date_range.days + 1
    
    metrics['overall_coverage_percent'] = (unique_days / expected_days) * 100 if expected_days > 0 else 0
    
    return metrics

def calculate_spatial_coverage(data: pd.DataFrame,
                               lat_column: str, 
                               lon_column: str,
                               region_boundaries: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Calculate spatial coverage metrics.
    
    Args:
        data: DataFrame to evaluate
        lat_column: Name of the column containing latitude
        lon_column: Name of the column containing longitude
        region_boundaries: Optional dictionary with region boundaries
            Format: {'min_lat': float, 'max_lat': float, 'min_lon': float, 'max_lon': float}
        
    Returns:
        Dict: Dictionary of spatial coverage metrics
    """
    metrics = {}
    
    # Clean data
    valid_coords = data.dropna(subset=[lat_column, lon_column])
    
    # Number of locations
    metrics['num_locations'] = len(valid_coords.drop_duplicates(subset=[lat_column, lon_column]))
    
    # Spatial distribution
    metrics['min_lat'] = valid_coords[lat_column].min()
    metrics['max_lat'] = valid_coords[lat_column].max()
    metrics['min_lon'] = valid_coords[lon_column].min()
    metrics['max_lon'] = valid_coords[lon_column].max()
    
    # Coverage area (simple estimate using rectangle)
    lat_range = metrics['max_lat'] - metrics['min_lat']
    lon_range = metrics['max_lon'] - metrics['min_lon']
    metrics['coverage_area_approx'] = lat_range * lon_range
    
    # If region boundaries provided, calculate coverage percentage
    if region_boundaries:
        region_lat_range = region_boundaries['max_lat'] - region_boundaries['min_lat']
        region_lon_range = region_boundaries['max_lon'] - region_boundaries['min_lon']
        region_area = region_lat_range * region_lon_range
        
        metrics['region_coverage_percent'] = (metrics['coverage_area_approx'] / region_area) * 100 if region_area > 0 else 0
    
    # Calculate density metrics
    if metrics['coverage_area_approx'] > 0:
        metrics['location_density'] = metrics['num_locations'] / metrics['coverage_area_approx']
    else:
        metrics['location_density'] = 0
    
    return metrics

def detect_outliers(data: pd.DataFrame, 
                    numeric_columns: Optional[List[str]] = None,
                    method: str = 'iqr',
                    threshold: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers in numeric columns.
    
    Args:
        data: DataFrame to evaluate
        numeric_columns: List of numeric column names to check. If None, all numeric columns are used.
        method: Method to use for outlier detection. Options: 'iqr', 'zscore'
        threshold: Threshold for outlier detection
            For IQR method: values outside Q1-threshold*IQR and Q3+threshold*IQR are outliers
            For zscore method: values with abs(zscore) > threshold are outliers
        
    Returns:
        Dict: Dictionary with outlier detection results
    """
    results = {}
    
    # If numeric columns not provided, identify them
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Process each column
    column_results = {}
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        values = data[column].dropna()
        if len(values) == 0:
            continue
            
        if method == 'iqr':
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            column_results[column] = {
                'count': len(outliers),
                'percent': (len(outliers) / len(values)) * 100 if len(values) > 0 else 0,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_value': values.min(),
                'max_value': values.max()
            }
            
        elif method == 'zscore':
            mean = values.mean()
            std = values.std()
            
            if std > 0:
                zscores = (values - mean) / std
                outliers = values[abs(zscores) > threshold]
                
                column_results[column] = {
                    'count': len(outliers),
                    'percent': (len(outliers) / len(values)) * 100 if len(values) > 0 else 0,
                    'zscore_threshold': threshold,
                    'min_value': values.min(),
                    'max_value': values.max()
                }
            else:
                column_results[column] = {
                    'count': 0,
                    'percent': 0,
                    'min_value': values.min(),
                    'max_value': values.max(),
                    'note': 'Standard deviation is zero, cannot calculate zscores'
                }
    
    results['column_outliers'] = column_results
    
    # Calculate overall statistics
    total_outliers = sum(col_result['count'] for col_result in column_results.values())
    total_values = sum(len(data[col].dropna()) for col in numeric_columns if col in data.columns)
    
    results['total_outliers'] = total_outliers
    results['total_outlier_percent'] = (total_outliers / total_values) * 100 if total_values > 0 else 0
    
    return results

def assess_temporal_consistency(data: pd.DataFrame,
                               date_column: str,
                               groupby_column: Optional[str] = None,
                               expected_frequency: str = 'D') -> Dict[str, Any]:
    """
    Assess the temporal consistency of time series data.
    
    Args:
        data: DataFrame to evaluate
        date_column: Name of the column containing dates/timestamps
        groupby_column: Optional column to group by (e.g., station_id)
        expected_frequency: Expected frequency of data, e.g., 'D' for daily, 'H' for hourly
        
    Returns:
        Dict: Dictionary with temporal consistency metrics
    """
    results = {}
    
    # Ensure datetime format
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    
    # If groupby provided, analyze each group
    if groupby_column:
        group_results = {}
        for group_name, group_data in data.groupby(groupby_column):
            # Sort by date
            group_data = group_data.sort_values(date_column)
            
            # Calculate time differences between consecutive records
            time_diffs = group_data[date_column].diff().dropna()
            
            # Convert to the expected frequency units
            if expected_frequency == 'D':
                time_diffs = time_diffs.dt.total_seconds() / (24 * 3600)  # Convert to days
                unit = 'days'
            elif expected_frequency == 'H':
                time_diffs = time_diffs.dt.total_seconds() / 3600  # Convert to hours
                unit = 'hours'
            elif expected_frequency == 'M':
                time_diffs = time_diffs.dt.days / 30  # Approximate months
                unit = 'months'
            else:
                time_diffs = time_diffs.dt.total_seconds()  # Seconds
                unit = 'seconds'
            
            # Calculate consistency metrics
            if len(time_diffs) > 0:
                group_result = {
                    'min_interval': time_diffs.min(),
                    'max_interval': time_diffs.max(),
                    'mean_interval': time_diffs.mean(),
                    'median_interval': time_diffs.median(),
                    'std_interval': time_diffs.std(),
                    'num_records': len(group_data),
                    'unit': unit
                }
                
                # Calculate gaps
                if expected_frequency == 'D':
                    expected_interval = 1.0  # 1 day
                elif expected_frequency == 'H':
                    expected_interval = 1.0  # 1 hour
                elif expected_frequency == 'M':
                    expected_interval = 1.0  # 1 month
                else:
                    expected_interval = 1.0  # Default, user should specify appropriate unit
                
                gaps = time_diffs[time_diffs > expected_interval * 1.5]  # 50% more than expected interval
                
                group_result['num_gaps'] = len(gaps)
                group_result['max_gap'] = gaps.max() if len(gaps) > 0 else 0
                group_result['gap_percent'] = (len(gaps) / len(time_diffs)) * 100 if len(time_diffs) > 0 else 0
                
                group_results[str(group_name)] = group_result
                
        results['group_consistency'] = group_results
        
        # Calculate average consistency across groups
        if group_results:
            results['avg_gap_percent'] = np.mean([g['gap_percent'] for g in group_results.values()])
    
    # Overall consistency
    data = data.sort_values(date_column)
    time_diffs = data[date_column].diff().dropna()
    
    # Convert to the expected frequency units
    if expected_frequency == 'D':
        time_diffs = time_diffs.dt.total_seconds() / (24 * 3600)  # Convert to days
        unit = 'days'
    elif expected_frequency == 'H':
        time_diffs = time_diffs.dt.total_seconds() / 3600  # Convert to hours
        unit = 'hours'
    elif expected_frequency == 'M':
        time_diffs = time_diffs.dt.days / 30  # Approximate months
        unit = 'months'
    else:
        time_diffs = time_diffs.dt.total_seconds()  # Seconds
        unit = 'seconds'
        
    # Calculate consistency metrics
    if len(time_diffs) > 0:
        results['overall_min_interval'] = time_diffs.min()
        results['overall_max_interval'] = time_diffs.max()
        results['overall_mean_interval'] = time_diffs.mean()
        results['overall_median_interval'] = time_diffs.median()
        results['overall_std_interval'] = time_diffs.std()
        results['overall_unit'] = unit
    
    return results

def evaluate_dataset_alignment(grid_data: pd.DataFrame, 
                              weather_data: pd.DataFrame,
                              outage_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate how well the different datasets align in time and space.
    
    Args:
        grid_data: DataFrame containing grid component data
        weather_data: DataFrame containing weather data
        outage_data: DataFrame containing outage data
        
    Returns:
        Dict: Dictionary with alignment metrics
    """
    results = {}
    
    # Check temporal alignment
    if 'timestamp' in weather_data.columns:
        weather_time_range = (
            weather_data['timestamp'].min(),
            weather_data['timestamp'].max()
        )
        results['weather_time_range'] = {
            'start': weather_time_range[0].strftime('%Y-%m-%d'),
            'end': weather_time_range[1].strftime('%Y-%m-%d')
        }
    
    outage_time_range = None
    if 'start_time' in outage_data.columns:
        outage_time_range = (
            outage_data['start_time'].min(),
            outage_data['start_time'].max() if 'end_time' not in outage_data.columns else outage_data['end_time'].max()
        )
        results['outage_time_range'] = {
            'start': outage_time_range[0].strftime('%Y-%m-%d'),
            'end': outage_time_range[1].strftime('%Y-%m-%d')
        }
    
    # Calculate temporal overlap
    if 'timestamp' in weather_data.columns and outage_time_range is not None:
        weather_start, weather_end = pd.to_datetime(weather_time_range[0]), pd.to_datetime(weather_time_range[1])
        outage_start, outage_end = pd.to_datetime(outage_time_range[0]), pd.to_datetime(outage_time_range[1])
        
        # Calculate overlap
        overlap_start = max(weather_start, outage_start)
        overlap_end = min(weather_end, outage_end)
        
        if overlap_end > overlap_start:
            overlap_days = (overlap_end - overlap_start).days
            weather_days = (weather_end - weather_start).days
            outage_days = (outage_end - outage_start).days
            
            results['temporal_overlap'] = {
                'overlap_days': overlap_days,
                'overlap_percent_of_weather': (overlap_days / weather_days) * 100 if weather_days > 0 else 0,
                'overlap_percent_of_outages': (overlap_days / outage_days) * 100 if outage_days > 0 else 0
            }
        else:
            results['temporal_overlap'] = {
                'overlap_days': 0,
                'note': 'No temporal overlap between weather and outage data'
            }
    
    # Check spatial alignment
    if all(col in grid_data.columns for col in ['location_x', 'location_y']) and all(col in weather_data.columns for col in ['latitude', 'longitude']):
        
        # Get grid component locations
        grid_locations = grid_data.dropna(subset=['location_x', 'location_y'])[['location_x', 'location_y']]
        
        # Calculate grid bounding box
        grid_min_x = grid_locations['location_x'].min() if not grid_locations.empty else None
        grid_max_x = grid_locations['location_x'].max() if not grid_locations.empty else None
        grid_min_y = grid_locations['location_y'].min() if not grid_locations.empty else None
        grid_max_y = grid_locations['location_y'].max() if not grid_locations.empty else None
        
        # Get weather station locations
        weather_locations = weather_data.dropna(subset=['latitude', 'longitude'])[['longitude', 'latitude']].drop_duplicates()
        
        # Calculate weather bounding box
        weather_min_x = weather_locations['longitude'].min() if not weather_locations.empty else None
        weather_max_x = weather_locations['longitude'].max() if not weather_locations.empty else None
        weather_min_y = weather_locations['latitude'].min() if not weather_locations.empty else None
        weather_max_y = weather_locations['latitude'].max() if not weather_locations.empty else None
        
        # Store bounding boxes
        if all(v is not None for v in [grid_min_x, grid_max_x, grid_min_y, grid_max_y]):
            results['grid_bounding_box'] = {
                'min_lon': grid_min_x,
                'max_lon': grid_max_x,
                'min_lat': grid_min_y,
                'max_lat': grid_max_y
            }
        
        if all(v is not None for v in [weather_min_x, weather_max_x, weather_min_y, weather_max_y]):
            results['weather_bounding_box'] = {
                'min_lon': weather_min_x,
                'max_lon': weather_max_x,
                'min_lat': weather_min_y,
                'max_lat': weather_max_y
            }
        
        # Calculate spatial overlap
        if all(v is not None for v in [grid_min_x, grid_max_x, grid_min_y, grid_max_y, 
                                     weather_min_x, weather_max_x, weather_min_y, weather_max_y]):
            
            # Calculate overlap
            overlap_min_x = max(grid_min_x, weather_min_x)
            overlap_max_x = min(grid_max_x, weather_max_x)
            overlap_min_y = max(grid_min_y, weather_min_y)
            overlap_max_y = min(grid_max_y, weather_max_y)
            
            if overlap_max_x > overlap_min_x and overlap_max_y > overlap_min_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                grid_area = (grid_max_x - grid_min_x) * (grid_max_y - grid_min_y)
                weather_area = (weather_max_x - weather_min_x) * (weather_max_y - weather_min_y)
                
                results['spatial_overlap'] = {
                    'overlap_area': overlap_area,
                    'overlap_percent_of_grid': (overlap_area / grid_area) * 100 if grid_area > 0 else 0,
                    'overlap_percent_of_weather': (overlap_area / weather_area) * 100 if weather_area > 0 else 0
                }
            else:
                results['spatial_overlap'] = {
                    'overlap_area': 0,
                    'note': 'No spatial overlap between grid and weather data'
                }
    
    # Check component coverage in outages
    if 'component_id' in grid_data.columns and 'component_id' in outage_data.columns:
        grid_components = set(grid_data['component_id'].unique())
        outage_components = set(outage_data['component_id'].unique())
        
        common_components = grid_components.intersection(outage_components)
        
        results['component_overlap'] = {
            'grid_components': len(grid_components),
            'outage_components': len(outage_components),
            'common_components': len(common_components),
            'percent_grid_with_outages': (len(common_components) / len(grid_components)) * 100 if len(grid_components) > 0 else 0,
            'percent_outages_with_grid': (len(common_components) / len(outage_components)) * 100 if len(outage_components) > 0 else 0
        }
    
    return results
