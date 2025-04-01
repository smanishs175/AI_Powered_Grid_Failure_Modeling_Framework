"""
Data validation utilities for the Data Management Module.

This module provides functions to validate different types of data
used in the Grid Failure Modeling Framework.
"""

import pandas as pd
import json
import logging
from typing import Dict, List, Union, Any

logger = logging.getLogger(__name__)

def validate_grid_topology_data(data: Dict[str, Any]) -> bool:
    """
    Validate grid topology data structure.
    
    Args:
        data: Dictionary containing grid topology data
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    # Check for required keys
    if not all(key in data for key in ['nodes', 'lines']):
        logger.error("Grid topology data missing required keys 'nodes' and/or 'lines'")
        return False
    
    # Validate nodes
    if not isinstance(data['nodes'], dict):
        logger.error("Grid topology 'nodes' must be a dictionary")
        return False
    
    # Check at least one node exists
    if len(data['nodes']) == 0:
        logger.error("Grid topology must contain at least one node")
        return False
    
    # Validate node structure
    for node_id, node in data['nodes'].items():
        if not isinstance(node, dict):
            logger.error(f"Node {node_id} data must be a dictionary")
            return False
        
        # Check node has an ID that matches its key
        if 'id' not in node:
            logger.error(f"Node {node_id} missing 'id' field")
            return False
        
        if node['id'] != node_id:
            logger.warning(f"Node {node_id} has mismatched ID in data: {node['id']}")
    
    # Validate lines
    if not isinstance(data['lines'], dict):
        logger.error("Grid topology 'lines' must be a dictionary")
        return False
    
    # Validate line structure
    for line_id, line in data['lines'].items():
        if not isinstance(line, dict):
            logger.error(f"Line {line_id} data must be a dictionary")
            return False
        
        # Check line has an ID that matches its key
        if 'id' not in line:
            logger.error(f"Line {line_id} missing 'id' field")
            return False
        
        if line['id'] != line_id:
            logger.warning(f"Line {line_id} has mismatched ID in data: {line['id']}")
        
        # Check line connects to valid nodes
        if 'from' not in line or 'to' not in line:
            logger.error(f"Line {line_id} missing 'from' or 'to' field")
            return False
        
        if line['from'] not in data['nodes']:
            logger.error(f"Line {line_id} references non-existent 'from' node: {line['from']}")
            return False
        
        if line['to'] not in data['nodes']:
            logger.error(f"Line {line_id} references non-existent 'to' node: {line['to']}")
            return False
    
    return True

def validate_weather_data(data: pd.DataFrame) -> bool:
    """
    Validate weather data.
    
    Args:
        data: DataFrame containing weather data
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    required_columns = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE']
    weather_columns = ['PRCP', 'SNOW', 'TMAX', 'TMIN', 'AWND']
    
    # Check for required columns
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        logger.error(f"Weather data missing required columns: {missing_required}")
        return False
    
    # Check for at least one weather measurement column
    available_weather = [col for col in weather_columns if col in data.columns]
    if not available_weather:
        logger.error(f"Weather data missing all weather measurement columns: {weather_columns}")
        return False
    
    # Check for date format
    try:
        pd.to_datetime(data['DATE'])
    except Exception as e:
        logger.error(f"Weather data contains invalid date formats: {e}")
        return False
    
    # Check for duplicate dates per station
    station_date_counts = data.groupby(['STATION', 'DATE']).size().reset_index(name='count')
    duplicates = station_date_counts[station_date_counts['count'] > 1]
    if not duplicates.empty:
        logger.warning(f"Weather data contains {len(duplicates)} duplicate station-date combinations")
    
    return True

def validate_outage_data(data: pd.DataFrame) -> bool:
    """
    Validate outage data.
    
    Args:
        data: DataFrame containing outage data
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    # Different outage datasets may have different column names
    # Let's handle both the eaglei_outages_*_merged.csv and eaglei_outages_*_agg.csv formats
    
    # For merged data
    required_merged_columns = ['fips', 'state', 'county', 'start_time', 'duration']
    
    # For aggregated data
    required_agg_columns = ['state', 'year', 'month', 'outage_count']
    
    # Check format type
    is_merged_format = all(col in data.columns for col in required_merged_columns)
    is_agg_format = all(col in data.columns for col in required_agg_columns)
    
    if not (is_merged_format or is_agg_format):
        logger.error("Outage data format not recognized. Missing required columns.")
        logger.error(f"For merged format, required: {required_merged_columns}")
        logger.error(f"For aggregated format, required: {required_agg_columns}")
        return False
    
    # Validate merged format
    if is_merged_format:
        # Check for valid start_time format
        try:
            pd.to_datetime(data['start_time'])
        except Exception as e:
            logger.error(f"Outage data contains invalid start_time formats: {e}")
            return False
        
        # Check for valid duration (should be numeric and non-negative)
        if not pd.api.types.is_numeric_dtype(data['duration']):
            logger.error("Outage data duration column is not numeric")
            return False
        
        if (data['duration'] < 0).any():
            logger.error("Outage data contains negative durations")
            return False
    
    # Validate aggregated format
    if is_agg_format:
        # Check for valid year and month
        if not pd.api.types.is_numeric_dtype(data['year']):
            logger.error("Outage data year column is not numeric")
            return False
        
        if not pd.api.types.is_numeric_dtype(data['month']):
            logger.error("Outage data month column is not numeric")
            return False
        
        if (data['month'] < 0).any() or (data['month'] > 11).any():
            logger.warning("Outage data contains month values outside 0-11 range")
        
        if (data['outage_count'] < 0).any():
            logger.error("Outage data contains negative outage counts")
            return False
    
    return True
