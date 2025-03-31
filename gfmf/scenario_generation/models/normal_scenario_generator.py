#!/usr/bin/env python
"""
Normal Scenario Generator

This module generates scenarios under normal operating conditions.
"""

import numpy as np
import pandas as pd
from .base_scenario_generator import BaseScenarioGenerator

class NormalScenarioGenerator(BaseScenarioGenerator):
    """
    Generator for normal operating condition scenarios.
    
    This class generates scenarios that represent typical failure patterns
    under normal operating conditions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the normal scenario generator.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        super().__init__(config)
        
        # Set default parameters if not provided in config
        self.normal_temp_range = config.get('normal_temp_range', (15, 30))
        self.normal_wind_range = config.get('normal_wind_range', (0, 15))
        self.normal_precip_range = config.get('normal_precip_range', (0, 5))
        self.normal_humidity_range = config.get('normal_humidity_range', (30, 70))
        
        # Proportion of time-related vs random failures
        self.time_related_proportion = config.get('time_related_proportion', 0.7)
        
    def generate_scenarios(self, input_data, count=50):
        """
        Generate normal operating condition scenarios.
        
        Args:
            input_data (dict): Dictionary containing input data.
            count (int): Number of scenarios to generate.
            
        Returns:
            list: List of generated normal scenario dictionaries.
        """
        self.logger.info(f"Generating {count} normal operating condition scenarios")
        
        # Extract required data
        components = input_data.get('components', pd.DataFrame())
        failure_probabilities = input_data.get('failure_probabilities', {})
        env_models = input_data.get('environmental_models', {})
        
        # Check that we have the necessary data
        if components.empty:
            self.logger.warning("No component data provided, cannot generate scenarios")
            return []
        
        # Generate scenarios
        scenarios = []
        
        for i in range(count):
            scenario_id = self.generate_scenario_id(prefix="normal")
            
            # Generate weather conditions
            weather_conditions = self._generate_normal_weather_conditions(env_models)
            
            # Select components to fail
            failure_count = np.random.randint(1, max(2, int(components.shape[0] * 0.05)))
            failed_components = self._select_components_to_fail(
                components, 
                failure_probabilities, 
                count=failure_count
            )
            
            # Generate failure times
            failure_times = self._generate_failure_times(len(failed_components))
            
            # Determine failure causes
            failure_causes = self._determine_failure_causes(
                failed_components, 
                components, 
                scenario_type='normal'
            )
            
            # Create component failures dictionary
            component_failures = {}
            for j, comp_id in enumerate(failed_components):
                component_failures[comp_id] = {
                    'failed': True,
                    'failure_time': failure_times[j],
                    'failure_cause': failure_causes[comp_id]
                }
            
            # Create scenario
            scenario = {
                'scenario_id': scenario_id,
                'weather_conditions': weather_conditions,
                'component_failures': component_failures,
                'metadata': self._create_scenario_metadata('statistical', 'normal')
            }
            
            scenarios.append(scenario)
            
        self.logger.info(f"Generated {len(scenarios)} normal scenarios")
        return scenarios
    
    def _generate_normal_weather_conditions(self, env_models=None):
        """
        Generate weather conditions for normal operating scenarios.
        
        Args:
            env_models (dict, optional): Environmental correlation models from Module 3.
            
        Returns:
            dict: Dictionary of weather conditions.
        """
        # Use correlation models if available
        if env_models and 'correlations' in env_models:
            # This would use sophisticated correlation models to generate realistic weather
            # For simplicity, we'll use random values within normal ranges
            pass
        
        # Generate random values within normal ranges
        temperature = np.random.uniform(*self.normal_temp_range)
        wind_speed = np.random.uniform(*self.normal_wind_range)
        precipitation = np.random.uniform(*self.normal_precip_range)
        humidity = np.random.uniform(*self.normal_humidity_range)
        pressure = np.random.uniform(980, 1020)  # Normal atmospheric pressure range
        
        # Add small random variations to make data more realistic
        weather_conditions = {
            'temperature': temperature,
            'wind_speed': wind_speed,
            'precipitation': precipitation,
            'humidity': humidity,
            'pressure': pressure,
            'is_extreme_temperature': False,
            'is_extreme_wind': False,
            'is_extreme_precipitation': False,
            'heat_wave_day': False,
            'cold_snap_day': False,
            'storm_day': False
        }
        
        return weather_conditions
