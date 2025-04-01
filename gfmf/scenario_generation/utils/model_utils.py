#!/usr/bin/env python
"""
Model Utilities for Scenario Generation

This module provides utility functions for the Scenario Generation Module.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

def setup_logger(name, log_level=logging.INFO):
    """
    Set up a logger with specified name and level.
    
    Args:
        name (str): Logger name.
        log_level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    
    # Only set up handler if not already configured
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    logger = logging.getLogger('model_utils')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return {}

def save_results(results, output_path, format='pickle'):
    """
    Save results to file.
    
    Args:
        results (object): Results to save.
        output_path (str): Path to save results to.
        format (str): Format to save results in ('pickle', 'json', 'csv').
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger = logging.getLogger('model_utils')
    
    try:
        # Create directory if it doesn't exist
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        if format == 'pickle':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
        elif format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format == 'csv' and isinstance(results, (pd.DataFrame, pd.Series)):
            results.to_csv(output_path, index=False)
        else:
            logger.warning(f"Unsupported format: {format}")
            return False
        
        logger.info(f"Saved results to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        return False

def generate_scenario_features(scenario):
    """
    Extract feature vector from a scenario for analysis.
    
    Args:
        scenario (dict): Scenario dictionary.
        
    Returns:
        np.ndarray: Feature vector.
    """
    # Extract key features from scenario
    features = []
    
    # Number of failed components
    features.append(len(scenario.get('component_failures', {})))
    
    # Weather conditions
    weather = scenario.get('weather_conditions', {})
    features.append(weather.get('temperature', 20))
    features.append(weather.get('wind_speed', 10))
    features.append(weather.get('precipitation', 0))
    features.append(weather.get('humidity', 50))
    
    # Extract extreme flags as binary features
    features.append(1 if weather.get('is_extreme_temperature', False) else 0)
    features.append(1 if weather.get('is_extreme_wind', False) else 0)
    features.append(1 if weather.get('is_extreme_precipitation', False) else 0)
    features.append(1 if weather.get('heat_wave_day', False) else 0)
    features.append(1 if weather.get('cold_snap_day', False) else 0)
    features.append(1 if weather.get('storm_day', False) else 0)
    
    # Failure timing features
    failure_times = [
        failure.get('failure_time', 0) 
        for failure in scenario.get('component_failures', {}).values()
    ]
    
    if failure_times:
        features.append(min(failure_times))  # First failure
        features.append(max(failure_times))  # Last failure
        features.append(np.mean(failure_times))  # Mean failure time
        features.append(np.std(failure_times) if len(failure_times) > 1 else 0)  # Time spread
    else:
        features.extend([0, 0, 0, 0])  # Default values if no failures
    
    return np.array(features)

def calculate_scenario_severity(scenario):
    """
    Calculate severity score for a scenario.
    
    Args:
        scenario (dict): Scenario dictionary.
        
    Returns:
        float: Severity score between 0 and 1.
    """
    # Count failed components
    num_failures = len(scenario.get('component_failures', {}))
    
    # Get weather extremity
    weather = scenario.get('weather_conditions', {})
    
    # Calculate base severity from failure count
    # Assume more than 10 failures is the worst case (severity=1)
    failure_severity = min(1.0, num_failures / 10.0)
    
    # Calculate weather severity
    weather_severity = 0.0
    
    # Check for extreme conditions
    if weather.get('is_extreme_temperature', False):
        weather_severity += 0.3
    if weather.get('is_extreme_wind', False):
        weather_severity += 0.3
    if weather.get('is_extreme_precipitation', False):
        weather_severity += 0.3
    
    # Cap weather severity at 1.0
    weather_severity = min(1.0, weather_severity)
    
    # Combined severity (weighted average)
    severity = (failure_severity * 0.7) + (weather_severity * 0.3)
    
    return severity

def validate_scenario_format(scenario):
    """
    Validate that a scenario has the correct format.
    
    Args:
        scenario (dict): Scenario dictionary.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    # Required top-level keys
    required_keys = ['scenario_id', 'weather_conditions', 'component_failures']
    
    if not all(key in scenario for key in required_keys):
        return False
    
    # Validate weather conditions
    weather = scenario.get('weather_conditions', {})
    weather_keys = ['temperature', 'wind_speed', 'precipitation']
    
    if not all(key in weather for key in weather_keys):
        return False
    
    # Validate component failures
    failures = scenario.get('component_failures', {})
    
    if not isinstance(failures, dict):
        return False
    
    # If there are failures, check their format
    for comp_id, failure in failures.items():
        if not isinstance(failure, dict):
            return False
        
        # Each failure should have these keys
        failure_keys = ['failed', 'failure_time', 'failure_cause']
        if not all(key in failure for key in failure_keys):
            return False
    
    return True
