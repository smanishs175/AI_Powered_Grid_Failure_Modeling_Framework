"""
Data Management Module for the Grid Failure Modeling Framework.

This module integrates data loading, preprocessing, and synthetic data generation
components into a unified interface.
"""

import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .data_loader import DataLoader
from .synthetic_generator import SyntheticGenerator
from .utils.validators import validate_grid_topology_data, validate_weather_data, validate_outage_data
from .utils.transformers import (
    transform_grid_topology_data, 
    transform_weather_data, 
    transform_outage_data, 
    align_datasets
)
from .utils.metrics import (
    calculate_data_completeness, 
    calculate_temporal_coverage,
    calculate_spatial_coverage,
    detect_outliers
)

# Configure logging
logger = logging.getLogger(__name__)

class DataManagementModule:
    """
    Main class for the Data Management Module.
    
    This class integrates data loading, preprocessing, and synthetic data generation
    and provides a unified interface for the Grid Failure Modeling Framework.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Data Management Module.
        
        Args:
            config_path: Path to the configuration file. If None, uses default config.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Create output directories
        self._create_directories()
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.synthetic_generator = SyntheticGenerator(self.config)
        
        # Initialize data storage
        self.data = {
            'grid': None,
            'weather': None,
            'outage': None,
            'combined': None
        }
        
        # Track data quality metrics
        self.data_quality = {
            'grid': {},
            'weather': {},
            'outage': {}
        }
        
        logger.info("Data Management Module initialized")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
            
        Returns:
            Dict: Configuration dictionary
        """
        # Default configuration
        default_config = {
            'data_paths': {
                'base_path': '',
                'grid_path': 'data/grid',
                'weather_path': 'data/weather',
                'outage_path': 'data/outage',
                'processed_path': 'data/processed',
                'synthetic_path': 'data/synthetic',
                'cache_path': 'data/cache'
            },
            'data_loading': {
                'weather_sample_limit': 10000,
                'validate_data': True
            },
            'data_preprocessing': {
                'remove_outliers': True,
                'standardize_weather': True,
                'impute_missing_values': True
            },
            'synthetic_data': {
                'num_nodes': 50,
                'num_lines': 75,
                'num_weather_stations': 5,
                'sim_start_date': '2023-01-01',
                'sim_end_date': '2023-12-31',
                'frequency': 'D',  # 'D' for daily, 'H' for hourly
                'num_outages': 100
            }
        }
        
        # If config file provided, load and merge with default
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                
                # Deep merge of nested dictionaries
                def deep_merge(d1, d2):
                    for k in d2:
                        if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
                            deep_merge(d1[k], d2[k])
                        else:
                            d1[k] = d2[k]
                
                deep_merge(default_config, user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading configuration from {config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def _create_directories(self) -> None:
        """Create necessary directories for data storage."""
        paths = self.config.get('data_paths', {})
        
        for path_name in ['processed_path', 'synthetic_path', 'cache_path']:
            if path_name in paths and paths[path_name]:
                # If it's a relative path, make it relative to the base path
                if not os.path.isabs(paths[path_name]) and 'base_path' in paths:
                    full_path = os.path.join(paths['base_path'], paths[path_name])
                else:
                    full_path = paths[path_name]
                
                os.makedirs(full_path, exist_ok=True)
                logger.debug(f"Created directory: {full_path}")
    
    def load_data(self, data_types: Optional[List[str]] = None, validate: bool = True) -> Dict[str, Any]:
        """
        Load data from specified sources.
        
        Args:
            data_types: List of data types to load ('grid', 'weather', 'outage').
                If None, load all types.
            validate: Whether to validate data after loading.
                
        Returns:
            Dict: Dictionary containing loaded data
        """
        # Load data
        loaded_data = self.data_loader.load_all(data_types)
        
        # Store data
        for data_type, data in loaded_data.items():
            self.data[data_type] = data
            
            # Calculate and store data quality metrics if validation is enabled
            if validate and data is not None:
                if data_type == 'grid':
                    if 'nodes' in data and 'lines' in data:
                        self.data_quality[data_type] = self._evaluate_grid_data_quality(data)
                elif data_type == 'weather':
                    self.data_quality[data_type] = self._evaluate_weather_data_quality(data)
                elif data_type == 'outage':
                    self.data_quality[data_type] = self._evaluate_outage_data_quality(data)
        
        logger.info(f"Loaded data types: {list(loaded_data.keys())}")
        return loaded_data
    
    def _evaluate_grid_data_quality(self, grid_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Evaluate quality of grid topology data.
        
        Args:
            grid_data: Dictionary containing grid data
            
        Returns:
            Dict: Data quality metrics
        """
        metrics = {}
        
        if 'nodes' in grid_data and not grid_data['nodes'].empty:
            nodes = grid_data['nodes']
            
            # Completeness
            metrics['node_completeness'] = calculate_data_completeness(nodes)
            
            # Spatial coverage
            if 'location_x' in nodes.columns and 'location_y' in nodes.columns:
                metrics['spatial_coverage'] = calculate_spatial_coverage(
                    nodes['location_x'],
                    nodes['location_y']
                )
        
        if 'lines' in grid_data and not grid_data['lines'].empty:
            lines = grid_data['lines']
            
            # Completeness
            metrics['line_completeness'] = calculate_data_completeness(lines)
            
            # Connectivity check
            if 'from_node' in lines.columns and 'to_node' in lines.columns:
                # Count unique connected nodes
                connected_nodes = set(lines['from_node'].unique()) | set(lines['to_node'].unique())
                metrics['connectivity'] = len(connected_nodes) / len(grid_data['nodes']) if 'nodes' in grid_data else 0
        
        return metrics
    
    def _evaluate_weather_data_quality(self, weather_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate quality of weather data.
        
        Args:
            weather_data: DataFrame containing weather data
            
        Returns:
            Dict: Data quality metrics
        """
        metrics = {}
        
        if weather_data is not None and not weather_data.empty:
            # Completeness
            metrics['completeness'] = calculate_data_completeness(weather_data)
            
            # Temporal coverage
            if 'timestamp' in weather_data.columns:
                metrics['temporal_coverage'] = calculate_temporal_coverage(weather_data['timestamp'])
            
            # Spatial coverage
            if 'latitude' in weather_data.columns and 'longitude' in weather_data.columns:
                metrics['spatial_coverage'] = calculate_spatial_coverage(
                    weather_data['longitude'],
                    weather_data['latitude']
                )
            
            # Outlier detection
            if 'temperature' in weather_data.columns:
                metrics['temperature_outliers'] = detect_outliers(weather_data['temperature'])
            
            if 'wind_speed' in weather_data.columns:
                metrics['wind_speed_outliers'] = detect_outliers(weather_data['wind_speed'])
            
            if 'precipitation' in weather_data.columns:
                metrics['precipitation_outliers'] = detect_outliers(weather_data['precipitation'])
        
        return metrics
    
    def _evaluate_outage_data_quality(self, outage_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate quality of outage data.
        
        Args:
            outage_data: DataFrame containing outage data
            
        Returns:
            Dict: Data quality metrics
        """
        metrics = {}
        
        if outage_data is not None and not outage_data.empty:
            # Completeness
            metrics['completeness'] = calculate_data_completeness(outage_data)
            
            # Temporal coverage
            if 'start_time' in outage_data.columns:
                metrics['temporal_coverage'] = calculate_temporal_coverage(outage_data['start_time'])
            
            # Outage duration outliers
            if 'duration' in outage_data.columns:
                metrics['duration_outliers'] = detect_outliers(outage_data['duration'])
        
        return metrics
    
    def preprocess_data(self, data_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Preprocess loaded data.
        
        Args:
            data_types: List of data types to preprocess ('grid', 'weather', 'outage').
                If None, preprocess all available types.
                
        Returns:
            Dict: Dictionary containing preprocessed data
        """
        if data_types is None:
            data_types = ['grid', 'weather', 'outage']
        
        # Check if data is loaded
        missing_types = [dtype for dtype in data_types if self.data[dtype] is None]
        if missing_types:
            logger.warning(f"Missing data for types: {missing_types}. Loading data first.")
            self.load_data(missing_types)
        
        # Preprocess each data type
        for data_type in data_types:
            if self.data[data_type] is not None:
                logger.info(f"Preprocessing {data_type} data")
                
                if data_type == 'grid':
                    # Transform grid data
                    self.data[data_type] = transform_grid_topology_data(
                        self.data[data_type], 
                        self.config.get('data_preprocessing', {})
                    )
                
                elif data_type == 'weather':
                    # Transform weather data
                    self.data[data_type] = transform_weather_data(
                        self.data[data_type], 
                        self.config.get('data_preprocessing', {})
                    )
                
                elif data_type == 'outage':
                    # Transform outage data
                    self.data[data_type] = transform_outage_data(
                        self.data[data_type], 
                        self.config.get('data_preprocessing', {})
                    )
        
        # Align datasets if we have all of them
        available_types = [dtype for dtype in ['grid', 'weather', 'outage'] if self.data[dtype] is not None]
        if set(available_types) == {'grid', 'weather', 'outage'}:
            logger.info("Aligning all datasets")
            self.data['combined'] = align_datasets(
                self.data['grid'], 
                self.data['weather'], 
                self.data['outage']
            )
        
        return self.data
    
    def generate_synthetic_data(self, 
                              grid_params: Optional[Dict[str, Any]] = None,
                              weather_params: Optional[Dict[str, Any]] = None,
                              outage_params: Optional[Dict[str, Any]] = None,
                              save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data.
        
        Args:
            grid_params: Parameters for grid data generation (overrides config)
            weather_params: Parameters for weather data generation (overrides config)
            outage_params: Parameters for outage data generation (overrides config)
            save: Whether to save the generated data to files
                
        Returns:
            Dict: Dictionary containing synthetic data DataFrames
        """
        logger.info("Generating synthetic data")
        
        # Get default parameters from config
        default_params = self.config.get('synthetic_data', {})
        
        # Merge with provided parameters
        if grid_params is None:
            grid_params = {}
        if weather_params is None:
            weather_params = {}
        if outage_params is None:
            outage_params = {}
        
        # Extract some common parameters from default
        num_nodes = grid_params.get('num_nodes', default_params.get('num_nodes', 50))
        num_lines = grid_params.get('num_lines', default_params.get('num_lines', 75))
        start_date = weather_params.get('start_date', default_params.get('sim_start_date', '2023-01-01'))
        end_date = weather_params.get('end_date', default_params.get('sim_end_date', '2023-12-31'))
        frequency = weather_params.get('frequency', default_params.get('frequency', 'D'))
        num_stations = weather_params.get('num_stations', default_params.get('num_weather_stations', 5))
        num_outages = outage_params.get('num_outages', default_params.get('num_outages', 100))
        
        # Generate grid data
        grid_data = self.synthetic_generator.generate_grid_topology(num_nodes=num_nodes, num_lines=num_lines)
        
        # Generate weather data
        weather_data = self.synthetic_generator.generate_weather_data(
            grid_data=grid_data,
            num_stations=num_stations,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency
        )
        
        # Generate outage data
        outage_data = self.synthetic_generator.generate_outage_data(
            grid_data=grid_data,
            weather_data=weather_data,
            num_outages=num_outages
        )
        
        # Generate combined data
        combined_data = self.synthetic_generator.generate_combined_data(
            grid_data=grid_data,
            weather_data=weather_data,
            outage_data=outage_data
        )
        
        # Store the generated data
        synthetic_data = {
            'grid': grid_data,
            'weather': weather_data,
            'outage': outage_data,
            'combined': combined_data
        }
        
        # Save to files if requested
        if save:
            save_path = self.config.get('data_paths', {}).get('synthetic_path', 'data/synthetic')
            
            # If it's a relative path, make it relative to the base path
            if not os.path.isabs(save_path) and 'base_path' in self.config.get('data_paths', {}):
                save_path = os.path.join(
                    self.config.get('data_paths', {}).get('base_path', ''),
                    save_path
                )
            
            # Generate a subfolder with timestamp to avoid overwriting
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(save_path, f"synthetic_{timestamp}")
            
            logger.info(f"Saving synthetic data to {save_path}")
            self.synthetic_generator._save_synthetic_data(synthetic_data, save_path)
        
        return synthetic_data
    
    def export_data(self, data_type: str, file_path: str, format: str = 'csv') -> bool:
        """
        Export data to a file.
        
        Args:
            data_type: Type of data to export ('grid', 'weather', 'outage', 'combined')
            file_path: Path to save the file
            format: Format to save the data in ('csv', 'json', 'pickle')
                
        Returns:
            bool: Success status
        """
        if data_type not in self.data or self.data[data_type] is None:
            logger.warning(f"No data available for type: {data_type}")
            return False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        data = self.data[data_type]
        
        try:
            if format.lower() == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, index=False)
                elif isinstance(data, dict) and 'nodes' in data and 'lines' in data:
                    # For grid data, export both nodes and lines
                    base_path = os.path.splitext(file_path)[0]
                    data['nodes'].to_csv(f"{base_path}_nodes.csv", index=False)
                    data['lines'].to_csv(f"{base_path}_lines.csv", index=False)
                else:
                    pd.DataFrame(data).to_csv(file_path, index=False)
            
            elif format.lower() == 'json':
                if isinstance(data, pd.DataFrame):
                    data.to_json(file_path, orient='records', date_format='iso')
                elif isinstance(data, dict):
                    with open(file_path, 'w') as f:
                        json.dump(data, f, default=str)
                else:
                    with open(file_path, 'w') as f:
                        json.dump(data, f, default=str)
            
            elif format.lower() == 'pickle':
                if isinstance(data, pd.DataFrame):
                    data.to_pickle(file_path)
                else:
                    import pickle
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
            
            else:
                logger.warning(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Exported {data_type} data to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def get_data_quality_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a report of data quality metrics for all data types.
        
        Returns:
            Dict: Dictionary containing data quality metrics
        """
        return self.data_quality
    
    def get_data(self, data_type: str) -> Any:
        """
        Get data of a specific type.
        
        Args:
            data_type: Type of data to retrieve ('grid', 'weather', 'outage', 'combined')
                
        Returns:
            Any: The requested data
        """
        return self.data.get(data_type)
