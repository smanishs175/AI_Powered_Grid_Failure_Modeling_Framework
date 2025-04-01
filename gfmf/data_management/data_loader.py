"""
Data loader classes for the Grid Failure Modeling Framework.

This module provides classes to load different types of data from various sources.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Any
import glob
from .utils.validators import validate_grid_topology_data, validate_weather_data, validate_outage_data

# Configure logging
logger = logging.getLogger(__name__)

class DataLoader:
    """Base data loader class for loading data of all types."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_path = config.get('data_paths', {}).get('base_path', '')
        
        # Initialize specific loaders
        self.grid_loader = GridTopologyLoader(config)
        self.weather_loader = WeatherDataLoader(config)
        self.outage_loader = OutageDataLoader(config)
    
    def load_all(self, data_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load all specified data types.
        
        Args:
            data_types: List of data types to load. Options: 'grid', 'weather', 'outage'.
                If None, load all types.
                
        Returns:
            Dict: Dictionary containing loaded data
        """
        if data_types is None:
            data_types = ['grid', 'weather', 'outage']
        
        result = {}
        
        if 'grid' in data_types:
            logger.info("Loading grid topology data")
            result['grid'] = self.grid_loader.load()
        
        if 'weather' in data_types:
            logger.info("Loading weather data")
            result['weather'] = self.weather_loader.load()
        
        if 'outage' in data_types:
            logger.info("Loading outage data")
            result['outage'] = self.outage_loader.load()
        
        return result


class GridTopologyLoader:
    """Loader for grid topology data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GridTopologyLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_path = config.get('data_paths', {}).get('base_path', '')
        self.grid_path = config.get('data_paths', {}).get('grid_path', '')
        self.cache_path = config.get('data_paths', {}).get('cache_path', '')
    
    def load(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load grid topology data.
        
        Args:
            file_path: Optional direct path to the file. If None, use the configured path.
                
        Returns:
            Dict: Dictionary containing loaded grid topology data
        """
        # Determine file path
        if file_path is None:
            search_paths = []
            
            # Check if specified grid path exists
            if self.grid_path:
                if os.path.isabs(self.grid_path):
                    search_paths.append(self.grid_path)
                else:
                    search_paths.append(os.path.join(self.base_path, self.grid_path))
            
            # Add default IEEE test case locations
            search_paths.extend([
                os.path.join(self.base_path, "data_collection_by_manish/IEEE Power System Test Cases"),
                os.path.join(self.base_path, "data_collection_by_manish/RTS_Data")
            ])
            
            # Find available grid files
            grid_files = []
            for path in search_paths:
                if os.path.exists(path):
                    # Look for JSON files in this path
                    json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
                    grid_files.extend(json_files)
            
            if not grid_files:
                logger.warning("No grid topology files found in search paths")
                return {}
            
            # For now, just use the first file found
            # In a real implementation, you might want more sophisticated selection logic
            file_path = grid_files[0]
            logger.info(f"Loading grid topology from {file_path}")
        
        try:
            # Load the file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate the data
            if not validate_grid_topology_data(data):
                logger.warning(f"Grid topology data validation failed for {file_path}")
            
            # Convert to DataFrames for easier processing
            nodes_df = pd.DataFrame.from_dict(data.get('nodes', {}), orient='index')
            lines_df = pd.DataFrame.from_dict(data.get('lines', {}), orient='index')
            
            return {
                'raw': data,
                'nodes': nodes_df,
                'lines': lines_df,
                'source_file': file_path
            }
            
        except Exception as e:
            logger.error(f"Error loading grid topology data: {e}")
            
            # Try to load from IEEE123 backup if needed
            if not file_path.endswith("IEEE123.json"):
                logger.info("Attempting to load backup IEEE123 test case")
                return self.load_ieee123_fallback()
            
            return {}
    
    def load_ieee123_fallback(self) -> Dict[str, Any]:
        """
        Load IEEE 123 bus test case as a fallback.
        
        Returns:
            Dict: Dictionary containing IEEE 123 bus test case data
        """
        # Create a simplified version of IEEE 123 bus test case
        nodes = {}
        lines = {}
        
        # Create 123 nodes
        for i in range(1, 124):
            node_id = f"node_{i}"
            nodes[node_id] = {
                "id": node_id,
                "type": "load" if i % 5 == 0 else "bus",
                "voltage": 4.16 if i < 100 else 13.2
            }
        
        # Create lines connecting nodes
        for i in range(1, 123):
            line_id = f"line_{i}"
            lines[line_id] = {
                "id": line_id,
                "from": f"node_{i}",
                "to": f"node_{i+1}",
                "length_km": 0.8,
                "type": "overhead"
            }
        
        # Add a few more lines to create loops
        for i in range(1, 120, 10):
            line_id = f"line_loop_{i}"
            lines[line_id] = {
                "id": line_id,
                "from": f"node_{i}",
                "to": f"node_{i+3}",
                "length_km": 1.2,
                "type": "overhead"
            }
        
        data = {
            "nodes": nodes,
            "lines": lines
        }
        
        # Convert to DataFrames
        nodes_df = pd.DataFrame.from_dict(data.get('nodes', {}), orient='index')
        lines_df = pd.DataFrame.from_dict(data.get('lines', {}), orient='index')
        
        return {
            'raw': data,
            'nodes': nodes_df,
            'lines': lines_df,
            'source_file': "IEEE123_fallback"
        }


class WeatherDataLoader:
    """Loader for weather data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the WeatherDataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_path = config.get('data_paths', {}).get('base_path', '')
        self.weather_path = config.get('data_paths', {}).get('weather_path', '')
        self.cache_path = config.get('data_paths', {}).get('cache_path', '')
        self.sample_limit = config.get('data_loading', {}).get('weather_sample_limit', 10000)
    
    def load(self, file_paths: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load weather data.
        
        Args:
            file_paths: Optional direct path(s) to file(s). If None, use the configured path.
                
        Returns:
            DataFrame: DataFrame containing loaded weather data
        """
        # Determine file paths
        if file_paths is None:
            search_paths = []
            
            # Check if specified weather path exists
            if self.weather_path:
                if os.path.isabs(self.weather_path):
                    search_paths.append(self.weather_path)
                else:
                    search_paths.append(os.path.join(self.base_path, self.weather_path))
            
            # Add default weather data locations
            search_paths.extend([
                os.path.join(self.base_path, "data_collection_by_manish/NOAA_Daily_Summaries_Reduced"),
                os.path.join(self.base_path, "data_collection_by_manish/Weather Data"),
                os.path.join(self.base_path, "data_collection_by_manish/Iowa Environmental Mesonet")
            ])
            
            # Find available weather files
            weather_files = []
            for path in search_paths:
                if os.path.exists(path):
                    # Look for CSV files in this path
                    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
                    weather_files.extend(csv_files)
            
            if not weather_files:
                logger.warning("No weather data files found in search paths")
                return pd.DataFrame()
            
            # Limit the number of files to process to avoid excessive memory usage
            file_paths = weather_files[:min(len(weather_files), 5)]
            logger.info(f"Loading weather data from {len(file_paths)} files")
        
        # Load and combine the data
        all_weather_data = []
        
        for file_path in file_paths:
            try:
                # Load the file
                logger.info(f"Loading weather data from {file_path}")
                data = pd.read_csv(file_path)
                
                # Sample to limit memory usage if needed
                if len(data) > self.sample_limit:
                    logger.info(f"Sampling {self.sample_limit} records from {len(data)} total")
                    data = data.sample(self.sample_limit, random_state=42)
                
                # Add file source column
                data['file_source'] = os.path.basename(file_path)
                
                # Validate the data
                if not validate_weather_data(data):
                    logger.warning(f"Weather data validation failed for {file_path}")
                
                all_weather_data.append(data)
                
            except Exception as e:
                logger.error(f"Error loading weather data from {file_path}: {e}")
        
        if not all_weather_data:
            logger.warning("No weather data loaded successfully")
            return pd.DataFrame()
        
        # Combine all datasets
        try:
            combined_data = pd.concat(all_weather_data, ignore_index=True)
            logger.info(f"Loaded {len(combined_data)} weather data records from {len(file_paths)} files")
            return combined_data
        except Exception as e:
            logger.error(f"Error combining weather data: {e}")
            # Return the first dataset if combination fails
            return all_weather_data[0] if all_weather_data else pd.DataFrame()


class OutageDataLoader:
    """Loader for outage data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OutageDataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_path = config.get('data_paths', {}).get('base_path', '')
        self.outage_path = config.get('data_paths', {}).get('outage_path', '')
        self.cache_path = config.get('data_paths', {}).get('cache_path', '')
    
    def load(self, file_paths: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load outage data.
        
        Args:
            file_paths: Optional direct path(s) to file(s). If None, use the configured path.
                
        Returns:
            DataFrame: DataFrame containing loaded outage data
        """
        # Determine file paths
        if file_paths is None:
            search_paths = []
            
            # Check if specified outage path exists
            if self.outage_path:
                if os.path.isabs(self.outage_path):
                    search_paths.append(self.outage_path)
                else:
                    search_paths.append(os.path.join(self.base_path, self.outage_path))
            
            # Add default outage data locations
            search_paths.extend([
                os.path.join(self.base_path, "data_collection_by_manish/Outage Data"),
                os.path.join(self.base_path, "data_collection_by_hollis/correlated_outage")
            ])
            
            # Find available outage files
            outage_files = []
            for path in search_paths:
                if os.path.exists(path):
                    # Look for CSV files in this path
                    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
                    outage_files.extend(csv_files)
            
            if not outage_files:
                logger.warning("No outage data files found in search paths")
                return pd.DataFrame()
            
            # Look for specific files we know work well
            preferred_files = [f for f in outage_files if 
                              ('eaglei_outages' in f.lower() or 
                               'outage_data' in f.lower())]
            
            if preferred_files:
                file_paths = preferred_files[:min(len(preferred_files), 3)]
            else:
                file_paths = outage_files[:min(len(outage_files), 3)]
                
            logger.info(f"Loading outage data from {len(file_paths)} files")
        
        # Load and combine the data
        all_outage_data = []
        
        for file_path in file_paths:
            try:
                # Load the file
                logger.info(f"Loading outage data from {file_path}")
                data = pd.read_csv(file_path)
                
                # Add file source column
                data['file_source'] = os.path.basename(file_path)
                
                # Validate the data
                if not validate_outage_data(data):
                    logger.warning(f"Outage data validation failed for {file_path}")
                
                all_outage_data.append(data)
                
            except Exception as e:
                logger.error(f"Error loading outage data from {file_path}: {e}")
        
        if not all_outage_data:
            logger.warning("No outage data loaded successfully")
            return pd.DataFrame()
        
        # Combine all datasets (if possible)
        try:
            # Check if the datasets can be concatenated
            # If they have different column structures, we may need more processing
            column_sets = [set(df.columns) for df in all_outage_data]
            common_columns = set.intersection(*column_sets) if column_sets else set()
            
            if len(common_columns) >= 3:
                # Enough common columns to concatenate
                combined_data = pd.concat(all_outage_data, ignore_index=True)
                logger.info(f"Loaded {len(combined_data)} outage data records from {len(file_paths)} files")
                return combined_data
            else:
                # Return the largest dataset
                largest_df = max(all_outage_data, key=len)
                logger.warning("Outage datasets have different structures, returning the largest dataset")
                return largest_df
                
        except Exception as e:
            logger.error(f"Error combining outage data: {e}")
            # Return the first dataset if combination fails
            return all_outage_data[0] if all_outage_data else pd.DataFrame()
