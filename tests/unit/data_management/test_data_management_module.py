"""
Unit tests for the Data Management Module (Module 1)
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gfmf.data_management.data_management_module import DataManagementModule


class TestDataManagementModule(unittest.TestCase):
    """Test cases for the DataManagementModule class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dm_module = DataManagementModule()
        
        # Create mock data
        self.grid_data = pd.DataFrame({
            'id': [1, 2, 3],
            'type': ['line', 'transformer', 'bus'],
            'capacity': [100, 200, 300],
            'age': [5, 10, 15]
        })
        
        self.weather_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'temperature': [10.5, 15.2, 12.8],
            'precipitation': [0, 10, 5],
            'humidity': [65, 80, 70],
            'wind_speed': [5, 15, 8]
        })
        
        self.outage_data = pd.DataFrame({
            'component_id': [1, 2, 3],
            'start_time': ['2023-01-01 10:00', '2023-01-02 12:00', '2023-01-03 09:00'],
            'end_time': ['2023-01-01 15:00', '2023-01-02 18:00', '2023-01-03 11:00'],
            'cause': ['weather', 'equipment failure', 'maintenance']
        })

    def test_initialization(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.dm_module)
        self.assertIsNotNone(self.dm_module.config)

    def test_load_data(self):
        """Test data loading functionality."""
        # Mock the DataLoader component
        self.dm_module.data_loader = MagicMock()
        self.dm_module.data_loader.load_all.return_value = {
            'grid': self.grid_data,
            'weather': self.weather_data,
            'outage': self.outage_data
        }
        
        # Mock the quality evaluation methods to avoid issues with internal implementation
        with patch.object(self.dm_module, '_evaluate_grid_data_quality', return_value={}):
            with patch.object(self.dm_module, '_evaluate_weather_data_quality', return_value={}):
                with patch.object(self.dm_module, '_evaluate_outage_data_quality', return_value={}):
                    # Call the load_data method
                    result = self.dm_module.load_data(['grid', 'weather', 'outage'])
                    
                    # Verify results
                    self.assertIsInstance(result, dict)
                    self.assertIn('grid', result)
                    self.assertIn('weather', result)
                    self.assertIn('outage', result)

    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        # Set up test data in the module
        self.dm_module.data = {
            'grid': self.grid_data,
            'weather': self.weather_data,
            'outage': self.outage_data,
            'combined': None
        }
        
        # Mock the transformer functions
        with patch('gfmf.data_management.data_management_module.transform_grid_topology_data', 
                  return_value=pd.DataFrame({'component_id': [1, 2, 3], 'type': ['line', 'transformer', 'bus']})):
            with patch('gfmf.data_management.data_management_module.transform_weather_data', 
                      return_value=self.weather_data):
                with patch('gfmf.data_management.data_management_module.transform_outage_data', 
                          return_value=self.outage_data):
                    with patch('gfmf.data_management.data_management_module.align_datasets', 
                              return_value=pd.DataFrame({'component_id': [1, 2, 3], 'date': ['2023-01-01', '2023-01-02', '2023-01-03']})):
                        # Call preprocess_data
                        result = self.dm_module.preprocess_data()
                        
                        # Verify results
                        self.assertIsInstance(result, dict)
                        self.assertIn('grid', result)
                        self.assertIn('weather', result)
                        self.assertIn('outage', result)
                        self.assertIn('combined', result)





    def test_data_quality_evaluation(self):
        """Test data quality evaluation functionality."""
        # Mock data
        grid_data = {
            'nodes': pd.DataFrame({
                'id': [1, 2, 3],
                'type': ['bus', 'bus', 'bus']
            }),
            'lines': pd.DataFrame({
                'id': [1, 2],
                'from': [1, 2],
                'to': [2, 3]
            })
        }
        
        # Set up test data
        self.dm_module.data = {
            'grid': grid_data,
            'weather': self.weather_data,
            'outage': self.outage_data
        }
        
        # Mock quality calculation functions
        with patch('gfmf.data_management.data_management_module.calculate_data_completeness', return_value=0.95):
            with patch('gfmf.data_management.data_management_module.calculate_temporal_coverage', return_value=0.9):
                with patch('gfmf.data_management.data_management_module.calculate_spatial_coverage', return_value=0.8):
                    with patch('gfmf.data_management.data_management_module.detect_outliers', return_value=[]):
                        # Test grid quality evaluation
                        grid_quality = self.dm_module._evaluate_grid_data_quality(grid_data)
                        self.assertIsInstance(grid_quality, dict)
                        
                        # Test weather quality evaluation
                        weather_quality = self.dm_module._evaluate_weather_data_quality(self.weather_data)
                        self.assertIsInstance(weather_quality, dict)
                        
                        # Test outage quality evaluation
                        outage_quality = self.dm_module._evaluate_outage_data_quality(self.outage_data)
                        self.assertIsInstance(outage_quality, dict)

    def test_generate_synthetic_data(self):
        """Test synthetic data generation functionality."""
        # Mock the SyntheticGenerator
        self.dm_module.synthetic_generator = MagicMock()
        self.dm_module.synthetic_generator.generate_grid_topology.return_value = {
            'nodes': pd.DataFrame({'id': [1, 2, 3]}),
            'lines': pd.DataFrame({'id': [1, 2], 'from': [1, 2], 'to': [2, 3]})
        }
        self.dm_module.synthetic_generator.generate_weather_data.return_value = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'temperature': [10.5, 15.2, 12.8]
        })
        self.dm_module.synthetic_generator.generate_outage_data.return_value = pd.DataFrame({
            'component_id': [1, 2, 3],
            'start_time': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        # Call generate_synthetic_data
        with patch('pandas.DataFrame.to_csv'):
            result = self.dm_module.generate_synthetic_data(save=True)
            
            # Verify results
            self.assertIsInstance(result, dict)
            self.assertIn('grid', result)
            self.assertIn('weather', result)
            self.assertIn('outage', result)


if __name__ == '__main__':
    unittest.main()
