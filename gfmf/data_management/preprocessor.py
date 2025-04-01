"""
Data preprocessing for the Grid Failure Modeling Framework.

This module provides the class for preprocessing and feature engineering.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Any
from .utils.transformers import (transform_grid_topology_data, transform_weather_data, 
                                transform_outage_data, align_datasets)
from .utils.metrics import (calculate_data_completeness, calculate_temporal_coverage,
                          calculate_spatial_coverage, detect_outliers,
                          assess_temporal_consistency, evaluate_dataset_alignment)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocessor for all data types in the Grid Failure Modeling Framework."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config
        self.base_path = config.get('data_paths', {}).get('base_path', '')
        self.processed_path = config.get('data_paths', {}).get('processed_path', 'data/processed')
        
        # Set default preprocessing options if not provided
        if 'preprocessing' not in self.config:
            self.config['preprocessing'] = {
                'missing_strategy': 'interpolate',
                'standardization': True,
                'outlier_handling': 'clip',
                'outlier_threshold': 3.0
            }
    
    def preprocess(self, raw_data: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Preprocess all data types in the raw data dictionary.
        
        Args:
            raw_data: Dictionary containing raw data loaded by DataLoader
            save_path: Optional path to save processed data
                
        Returns:
            Dict: Dictionary containing processed data
        """
        processed_data = {}
        quality_metrics = {}
        
        # Preprocess grid data if available
        if 'grid' in raw_data and raw_data['grid']:
            logger.info("Preprocessing grid data")
            try:
                if isinstance(raw_data['grid'], dict) and 'nodes' in raw_data['grid'] and 'lines' in raw_data['grid']:
                    # Transform grid data
                    processed_data['grid'] = transform_grid_topology_data(
                        raw_data['grid'],
                        self.config['preprocessing'] if 'preprocessing' in self.config else None
                    )
                    
                    # Calculate quality metrics
                    quality_metrics['grid'] = {
                        'completeness': calculate_data_completeness(processed_data['grid'])
                    }
                    
                    logger.info(f"Processed grid data with {len(processed_data['grid'])} components")
                else:
                    logger.warning("Grid data not in expected format, skipping preprocessing")
            except Exception as e:
                logger.error(f"Error preprocessing grid data: {e}")
                # Initialize with empty DataFrame to avoid errors in downstream processing
                processed_data['grid'] = pd.DataFrame()
        else:
            logger.warning("No grid data available for preprocessing")
            # Initialize with empty DataFrame to avoid errors in downstream processing
            processed_data['grid'] = pd.DataFrame()
        
        # Preprocess weather data if available
        if 'weather' in raw_data and not raw_data['weather'].empty:
            logger.info("Preprocessing weather data")
            try:
                # Transform weather data
                processed_data['weather'] = transform_weather_data(
                    raw_data['weather'],
                    self.config
                )
                
                # Calculate quality metrics
                quality_metrics['weather'] = {
                    'completeness': calculate_data_completeness(processed_data['weather']),
                    'temporal_coverage': calculate_temporal_coverage(
                        processed_data['weather'], 
                        'timestamp',
                        'station_id' if 'station_id' in processed_data['weather'].columns else None
                    ),
                    'outliers': detect_outliers(
                        processed_data['weather'],
                        method=self.config['preprocessing'].get('outlier_method', 'iqr'),
                        threshold=self.config['preprocessing'].get('outlier_threshold', 1.5)
                    ),
                    'temporal_consistency': assess_temporal_consistency(
                        processed_data['weather'],
                        'timestamp',
                        'station_id' if 'station_id' in processed_data['weather'].columns else None,
                        expected_frequency='D'
                    )
                }
                
                # Handle outliers if specified
                outlier_handling = self.config['preprocessing'].get('outlier_handling', 'none')
                if outlier_handling == 'clip':
                    # Clip outliers
                    numeric_columns = processed_data['weather'].select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_columns:
                        q1 = processed_data['weather'][col].quantile(0.25)
                        q3 = processed_data['weather'][col].quantile(0.75)
                        iqr = q3 - q1
                        
                        lower_bound = q1 - self.config['preprocessing'].get('outlier_threshold', 1.5) * iqr
                        upper_bound = q3 + self.config['preprocessing'].get('outlier_threshold', 1.5) * iqr
                        
                        processed_data['weather'][col] = processed_data['weather'][col].clip(lower_bound, upper_bound)
                
                logger.info(f"Processed weather data with {len(processed_data['weather'])} records")
            except Exception as e:
                logger.error(f"Error preprocessing weather data: {e}")
                # Initialize with empty DataFrame to avoid errors in downstream processing
                processed_data['weather'] = pd.DataFrame()
        else:
            logger.warning("No weather data available for preprocessing")
            # Initialize with empty DataFrame to avoid errors in downstream processing
            processed_data['weather'] = pd.DataFrame()
        
        # Preprocess outage data if available
        if 'outage' in raw_data and not raw_data['outage'].empty:
            logger.info("Preprocessing outage data")
            try:
                # Transform outage data
                processed_data['outage'] = transform_outage_data(
                    raw_data['outage'],
                    self.config
                )
                
                # Calculate quality metrics
                quality_metrics['outage'] = {
                    'completeness': calculate_data_completeness(processed_data['outage'])
                }
                
                # Add additional metrics if required fields are present
                if 'start_time' in processed_data['outage'].columns:
                    quality_metrics['outage']['temporal_coverage'] = calculate_temporal_coverage(
                        processed_data['outage'], 
                        'start_time',
                        'component_id' if 'component_id' in processed_data['outage'].columns else None
                    )
                    
                    quality_metrics['outage']['temporal_consistency'] = assess_temporal_consistency(
                        processed_data['outage'],
                        'start_time',
                        'component_id' if 'component_id' in processed_data['outage'].columns else None,
                        expected_frequency='D'
                    )
                
                logger.info(f"Processed outage data with {len(processed_data['outage'])} records")
            except Exception as e:
                logger.error(f"Error preprocessing outage data: {e}")
                # Initialize with empty DataFrame to avoid errors in downstream processing
                processed_data['outage'] = pd.DataFrame()
        else:
            logger.warning("No outage data available for preprocessing")
            # Initialize with empty DataFrame to avoid errors in downstream processing
            processed_data['outage'] = pd.DataFrame()
        
        # Create combined dataset if all required data is available
        grid_available = 'grid' in processed_data and not processed_data['grid'].empty
        weather_available = 'weather' in processed_data and not processed_data['weather'].empty
        outage_available = 'outage' in processed_data and not processed_data['outage'].empty
        
        if grid_available and weather_available and outage_available:
            logger.info("Creating combined dataset")
            try:
                processed_data['combined'] = align_datasets(
                    processed_data['grid'],
                    processed_data['weather'],
                    processed_data['outage'],
                    self.config['preprocessing'] if 'preprocessing' in self.config else None
                )
                
                # Calculate alignment metrics
                quality_metrics['combined'] = {
                    'alignment': evaluate_dataset_alignment(
                        processed_data['grid'],
                        processed_data['weather'],
                        processed_data['outage']
                    ),
                    'completeness': calculate_data_completeness(processed_data['combined'])
                }
                
                logger.info(f"Created combined dataset with {len(processed_data['combined'])} records")
            except Exception as e:
                logger.error(f"Error creating combined dataset: {e}")
                processed_data['combined'] = pd.DataFrame()
        else:
            logger.warning("Not all required data available for creating combined dataset")
            processed_data['combined'] = pd.DataFrame()
        
        # Save processed data if save_path is provided
        if save_path:
            self._save_processed_data(processed_data, save_path, quality_metrics)
        
        # Add quality metrics to the result
        processed_data['quality_metrics'] = quality_metrics
        
        return processed_data
    
    def _save_processed_data(self, processed_data: Dict[str, Any], 
                            save_path: str,
                            quality_metrics: Dict[str, Any]) -> None:
        """
        Save processed data to disk.
        
        Args:
            processed_data: Dictionary containing processed data
            save_path: Path to save processed data
            quality_metrics: Dictionary containing data quality metrics
        """
        try:
            # Create the save directory if it doesn't exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                logger.info(f"Created directory: {save_path}")
            
            # Save each dataset
            for key, data in processed_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Determine file path
                    file_path = os.path.join(save_path, f"{key}_data.csv")
                    
                    # Save the data
                    data.to_csv(file_path, index=False)
                    logger.info(f"Saved {key} data to {file_path}")
            
            # Save quality metrics as JSON
            import json
            metrics_path = os.path.join(save_path, "quality_metrics.json")
            
            # Convert any non-serializable objects
            def clean_for_json(obj):
                if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
                    return str(obj)
                if isinstance(obj, (np.int64, np.float64)):
                    return int(obj) if isinstance(obj, np.int64) else float(obj)
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean_for_json(v) for v in obj]
                return obj
            
            clean_metrics = clean_for_json(quality_metrics)
            
            with open(metrics_path, 'w') as f:
                json.dump(clean_metrics, f, indent=2)
            
            logger.info(f"Saved quality metrics to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def preprocess_grid_data(self, grid_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess grid topology data.
        
        Args:
            grid_data: Dictionary containing grid data from GridTopologyLoader
                
        Returns:
            DataFrame: DataFrame containing processed grid data
        """
        if not grid_data or not isinstance(grid_data, dict) or 'nodes' not in grid_data or 'lines' not in grid_data:
            logger.warning("Invalid grid data provided for preprocessing")
            return pd.DataFrame()
        
        try:
            # Transform grid data
            processed_grid = transform_grid_data(
                grid_data['nodes'],
                grid_data['lines']
            )
            
            logger.info(f"Processed grid data with {len(processed_grid)} components")
            return processed_grid
            
        except Exception as e:
            logger.error(f"Error preprocessing grid data: {e}")
            return pd.DataFrame()
    
    def preprocess_weather_data(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess weather data.
        
        Args:
            weather_data: DataFrame containing weather data from WeatherDataLoader
                
        Returns:
            DataFrame: DataFrame containing processed weather data
        """
        if weather_data is None or weather_data.empty:
            logger.warning("No weather data provided for preprocessing")
            return pd.DataFrame()
        
        try:
            # Transform weather data
            processed_weather = transform_weather_data(
                weather_data,
                self.config
            )
            
            # Handle outliers if specified
            outlier_handling = self.config['preprocessing'].get('outlier_handling', 'none')
            if outlier_handling == 'clip':
                # Clip outliers
                numeric_columns = processed_weather.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_columns:
                    q1 = processed_weather[col].quantile(0.25)
                    q3 = processed_weather[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - self.config['preprocessing'].get('outlier_threshold', 1.5) * iqr
                    upper_bound = q3 + self.config['preprocessing'].get('outlier_threshold', 1.5) * iqr
                    
                    processed_weather[col] = processed_weather[col].clip(lower_bound, upper_bound)
            
            logger.info(f"Processed weather data with {len(processed_weather)} records")
            return processed_weather
            
        except Exception as e:
            logger.error(f"Error preprocessing weather data: {e}")
            return pd.DataFrame()
    
    def preprocess_outage_data(self, outage_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess outage data.
        
        Args:
            outage_data: DataFrame containing outage data from OutageDataLoader
                
        Returns:
            DataFrame: DataFrame containing processed outage data
        """
        if outage_data is None or outage_data.empty:
            logger.warning("No outage data provided for preprocessing")
            return pd.DataFrame()
        
        try:
            # Transform outage data
            processed_outage = transform_outage_data(
                outage_data,
                self.config
            )
            
            logger.info(f"Processed outage data with {len(processed_outage)} records")
            return processed_outage
            
        except Exception as e:
            logger.error(f"Error preprocessing outage data: {e}")
            return pd.DataFrame()
