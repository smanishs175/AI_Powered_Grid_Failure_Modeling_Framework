#!/usr/bin/env python
"""
Failure Prediction Data Loader

This module loads data from the Failure Prediction Module (Module 3)
for use in scenario generation.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

class FailurePredictionDataLoader:
    """
    Loads and processes data from the Failure Prediction Module.
    
    This class handles loading prediction models, failure probabilities,
    and other outputs from the Failure Prediction Module.
    """
    
    def __init__(self, config=None):
        """
        Initialize the data loader.
        
        Args:
            config (dict, optional): Configuration dictionary with data paths.
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set default paths if not provided in config
        self.base_path = self.config.get('base_path', 'data/failure_prediction/')
        self.probabilities_path = self.config.get('probabilities_path', 
                                            os.path.join(self.base_path, 'failure_probabilities.pkl'))
        self.time_series_path = self.config.get('time_series_path', 
                                          os.path.join(self.base_path, 'time_series_forecasts.pkl'))
        self.extreme_event_path = self.config.get('extreme_event_path', 
                                            os.path.join(self.base_path, 'extreme_event_impacts.pkl'))
        self.correlation_path = self.config.get('correlation_path', 
                                          os.path.join(self.base_path, 'correlation_models.pkl'))
                                          
        # Paths for component and outage data
        self.components_path = self.config.get('components_path', 'data/synthetic/synthetic_20250328_144932/synthetic_grid.csv')
        self.outages_path = self.config.get('outages_path', 'data/synthetic/synthetic_20250328_144932/synthetic_outages.csv')
        self.weather_path = self.config.get('weather_path', 'data/synthetic/synthetic_20250328_144932/synthetic_weather.csv')

    def load_data(self, use_synthetic=False):
        """
        Load data from the Failure Prediction Module.
        
        Args:
            use_synthetic (bool): If True, generate synthetic data instead of loading from files.
            
        Returns:
            dict: Dictionary containing all loaded data.
        """
        self.logger.info("Loading data from Failure Prediction Module")
        
        # Container for all loaded data
        input_data = {}
        
        # Load component data
        input_data['components'] = self._load_component_data()
        
        # Load outage records (if available)
        input_data['outage_records'] = self._load_outage_data()
        
        if use_synthetic:
            self.logger.info("Generating synthetic prediction data")
            
            # Generate synthetic prediction data
            input_data.update(self._generate_synthetic_prediction_data(input_data['components']))
        else:
            # Load failure probabilities
            input_data['failure_probabilities'] = self._load_failure_probabilities()
            
            # Load time series forecasts
            input_data['time_series_forecasts'] = self._load_time_series_forecasts()
            
            # Load extreme event impacts
            input_data['extreme_event_models'] = self._load_extreme_event_impacts()
            
            # Load environmental correlation models
            input_data['environmental_models'] = self._load_correlation_models()
        
        self.logger.info("Data loading complete")
        return input_data

    def _load_component_data(self):
        """
        Load component data from CSV file.
        
        Returns:
            DataFrame: DataFrame containing component data.
        """
        try:
            components = pd.read_csv(self.components_path)
            self.logger.info(f"Loaded {len(components)} components from {self.components_path}")
            return components
        except Exception as e:
            self.logger.warning(f"Error loading component data: {e}")
            self.logger.info("Returning empty DataFrame for components")
            return pd.DataFrame()

    def _load_outage_data(self):
        """
        Load outage records from CSV file.
        
        Returns:
            DataFrame: DataFrame containing outage records.
        """
        try:
            outages = pd.read_csv(self.outages_path)
            
            # Convert timestamp columns to datetime
            timestamp_cols = ['start_time', 'end_time']
            for col in timestamp_cols:
                if col in outages.columns:
                    outages[col] = pd.to_datetime(outages[col])
            
            self.logger.info(f"Loaded {len(outages)} outage records from {self.outages_path}")
            return outages
        except Exception as e:
            self.logger.warning(f"Error loading outage data: {e}")
            self.logger.info("Returning empty DataFrame for outages")
            return pd.DataFrame()

    def _load_failure_probabilities(self):
        """
        Load component failure probabilities from pickle file.
        
        Returns:
            dict: Dictionary mapping component IDs to failure probabilities.
        """
        try:
            with open(self.probabilities_path, 'rb') as f:
                probabilities = pickle.load(f)
            
            self.logger.info(f"Loaded failure probabilities for {len(probabilities)} components")
            return probabilities
        except Exception as e:
            self.logger.warning(f"Error loading failure probabilities: {e}")
            self.logger.info("Generating default failure probabilities")
            
            # Generate default probabilities from component data
            components = self._load_component_data()
            if not components.empty and 'component_id' in components.columns:
                return self._generate_default_probabilities(components)
            else:
                return {}

    def _load_time_series_forecasts(self):
        """
        Load time series forecasts from pickle file.
        
        Returns:
            dict: Dictionary containing time series forecasting models and results.
        """
        try:
            with open(self.time_series_path, 'rb') as f:
                forecasts = pickle.load(f)
            
            self.logger.info(f"Loaded time series forecasts")
            return forecasts
        except Exception as e:
            self.logger.warning(f"Error loading time series forecasts: {e}")
            return {}

    def _load_extreme_event_impacts(self):
        """
        Load extreme event impact models from pickle file.
        
        Returns:
            dict: Dictionary mapping event types to impact models.
        """
        try:
            with open(self.extreme_event_path, 'rb') as f:
                impact_models = pickle.load(f)
            
            self.logger.info(f"Loaded extreme event impact models for {len(impact_models)} event types")
            return impact_models
        except Exception as e:
            self.logger.warning(f"Error loading extreme event impact models: {e}")
            return {}

    def _load_correlation_models(self):
        """
        Load environmental correlation models from pickle file.
        
        Returns:
            dict: Dictionary containing correlation models and results.
        """
        try:
            with open(self.correlation_path, 'rb') as f:
                correlation_models = pickle.load(f)
            
            self.logger.info(f"Loaded environmental correlation models")
            return correlation_models
        except Exception as e:
            self.logger.warning(f"Error loading correlation models: {e}")
            return {}

    def _generate_synthetic_prediction_data(self, components):
        """
        Generate synthetic prediction data for testing.
        
        Args:
            components (DataFrame): DataFrame containing component information.
            
        Returns:
            dict: Dictionary containing synthetic prediction data.
        """
        self.logger.info("Generating synthetic prediction data")
        
        # Container for synthetic data
        synthetic_data = {}
        
        # Generate synthetic failure probabilities
        failure_probabilities = {}
        
        if not components.empty and 'component_id' in components.columns:
            for _, component in components.iterrows():
                comp_id = component['component_id']
                
                # Generate random probability weighted by component age if available
                if 'age' in component:
                    age_factor = min(1.0, component['age'] / 50.0) * 0.5
                    base_prob = 0.01 + (age_factor * 0.09)  # 0.01 to 0.1 based on age
                else:
                    base_prob = 0.05  # Default
                
                # Add some randomness
                prob = min(0.95, max(0.001, base_prob * (0.5 + np.random.random())))
                
                failure_probabilities[comp_id] = prob
        
        synthetic_data['failure_probabilities'] = failure_probabilities
        
        # Generate synthetic extreme event impact models
        synthetic_data['extreme_event_models'] = {
            'high_temperature': {'impact_factor': 2.5},
            'low_temperature': {'impact_factor': 2.0},
            'high_wind': {'impact_factor': 3.0},
            'precipitation': {'impact_factor': 2.0}
        }
        
        # Generate synthetic environmental correlation models
        synthetic_data['environmental_models'] = {
            'correlations': {
                'temperature': {'wind_speed': -0.2, 'precipitation': 0.3},
                'wind_speed': {'temperature': -0.2, 'precipitation': 0.4},
                'precipitation': {'temperature': 0.3, 'wind_speed': 0.4}
            }
        }
        
        # Generate synthetic time series forecasts
        synthetic_data['time_series_forecasts'] = {
            'forecast_horizon': 48,
            'confidence_intervals': {'lower': 0.9, 'upper': 1.1}
        }
        
        self.logger.info("Generated synthetic prediction data")
        return synthetic_data

    def _generate_default_probabilities(self, components):
        """
        Generate default failure probabilities based on component data.
        
        Args:
            components (DataFrame): DataFrame containing component information.
            
        Returns:
            dict: Dictionary mapping component IDs to default failure probabilities.
        """
        self.logger.info("Generating default failure probabilities")
        
        probabilities = {}
        
        # Define default probability based on component type and age
        type_base_probs = {
            'transformer': 0.01,
            'line': 0.02,
            'generator': 0.015,
            'substation': 0.005,
            'switch': 0.01,
            'default': 0.01
        }
        
        # Age factors (multipliers for base probability)
        age_factors = {
            (0, 5): 0.5,     # 0-5 years: half the base probability
            (5, 15): 1.0,    # 5-15 years: base probability
            (15, 30): 1.5,   # 15-30 years: 1.5x base probability
            (30, float('inf')): 2.5  # >30 years: 2.5x base probability
        }
        
        for _, component in components.iterrows():
            comp_id = component['component_id']
            comp_type = component.get('type', 'default').lower()
            comp_age = component.get('age', 15)
            
            # Get base probability for component type
            base_prob = type_base_probs.get(comp_type, type_base_probs['default'])
            
            # Apply age factor
            age_factor = next(
                (factor for (min_age, max_age), factor in age_factors.items() 
                 if min_age <= comp_age < max_age), 
                age_factors[(30, float('inf'))]
            )
            
            # Calculate final probability
            final_prob = base_prob * age_factor
            
            # Add some random variation (±10%)
            variation = 0.9 + np.random.random() * 0.2  # 0.9 to 1.1
            final_prob *= variation
            
            # Ensure probability is within [0,1]
            probabilities[comp_id] = max(0, min(1, final_prob))
        
        self.logger.info(f"Generated default probabilities for {len(probabilities)} components")
        return probabilities
