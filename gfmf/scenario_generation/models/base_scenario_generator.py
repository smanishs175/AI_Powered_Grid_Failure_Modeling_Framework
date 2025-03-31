#!/usr/bin/env python
"""
Base Scenario Generator

This module defines the base class for all scenario generators.
"""

import uuid
import logging
import datetime
import numpy as np
import pandas as pd

class BaseScenarioGenerator:
    """
    Base class for scenario generators.
    
    This class provides common functionality for all types of scenario generators.
    """
    
    def __init__(self, config=None):
        """
        Initialize the base scenario generator.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_scenario_id(self, prefix="scenario"):
        """
        Generate a unique scenario ID.
        
        Args:
            prefix (str): Prefix for the scenario ID.
            
        Returns:
            str: Unique scenario ID.
        """
        # Generate a timestamped UUID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_id}"
    
    def _select_components_to_fail(self, components, failure_probabilities, count=None, max_percentage=0.1):
        """
        Select components that will fail in a scenario based on failure probabilities.
        
        Args:
            components (DataFrame): DataFrame containing component information.
            failure_probabilities (dict): Dictionary mapping component IDs to failure probabilities.
            count (int, optional): Number of components to select. If None, uses failure probabilities.
            max_percentage (float): Maximum percentage of components that can fail.
            
        Returns:
            list: List of component IDs that will fail.
        """
        # Validate inputs
        if not components.shape[0]:
            self.logger.warning("No components provided, returning empty list")
            return []
        
        # Determine maximum number of components to fail
        max_components = max(1, int(components.shape[0] * max_percentage))
        
        if count is not None:
            # Fixed count mode
            count = min(count, max_components)
            
            # Convert failure probabilities to series for weighted sampling
            comp_ids = list(components['component_id'])
            weights = [failure_probabilities.get(comp_id, 0.01) for comp_id in comp_ids]
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Sample components with replacement (to handle potential duplicates)
            selected_indices = np.random.choice(
                range(len(comp_ids)), 
                size=min(count * 2, len(comp_ids)), 
                replace=False, 
                p=weights
            )
            
            # Take unique components up to the count
            selected_components = [comp_ids[i] for i in selected_indices]
            return selected_components[:count]
        else:
            # Probability-based mode
            # Select components based on their failure probability
            failed_components = []
            
            for _, component in components.iterrows():
                comp_id = component['component_id']
                prob = failure_probabilities.get(comp_id, 0.01)
                
                if np.random.random() < prob:
                    failed_components.append(comp_id)
            
            # Limit the number of failures to maximum percentage
            if len(failed_components) > max_components:
                # If too many components selected, subsample
                failed_components = np.random.choice(
                    failed_components, 
                    size=max_components, 
                    replace=False
                ).tolist()
            
            return failed_components
    
    def _generate_failure_times(self, num_failures, time_window=24.0):
        """
        Generate timestamps for component failures within a time window.
        
        Args:
            num_failures (int): Number of failures to generate times for.
            time_window (float): Time window in hours.
            
        Returns:
            list: List of failure times (hours from start).
        """
        # Generate random times within the window
        return sorted(np.random.uniform(0, time_window, num_failures))
    
    def _determine_failure_causes(self, component_ids, components_df, scenario_type='normal'):
        """
        Determine the causes of failures for each component.
        
        Args:
            component_ids (list): List of component IDs that failed.
            components_df (DataFrame): DataFrame containing component information.
            scenario_type (str): Type of scenario ('normal', 'high_temperature', etc.)
            
        Returns:
            dict: Dictionary mapping component IDs to failure causes.
        """
        # Map of scenario types to most likely failure causes
        scenario_causes = {
            'normal': ['equipment_failure', 'aging', 'maintenance_issue', 'unknown'],
            'high_temperature': ['overheating', 'thermal_stress', 'equipment_failure'],
            'low_temperature': ['freezing', 'mechanical_stress', 'equipment_failure'],
            'high_wind': ['wind_damage', 'physical_damage', 'equipment_failure'],
            'precipitation': ['water_damage', 'flooding', 'equipment_failure'],
            'compound': ['multiple_stressors', 'cascading_failure', 'equipment_failure']
        }
        
        # Get causes for this scenario type
        causes = scenario_causes.get(scenario_type, scenario_causes['normal'])
        
        # Assign causes to each component, with some probability distribution
        failure_causes = {}
        
        # Main cause is more likely
        probabilities = [0.5] + [0.5 / (len(causes) - 1)] * (len(causes) - 1)
        
        for comp_id in component_ids:
            # Get component type for more specific cause assignment
            comp_info = components_df[components_df['component_id'] == comp_id]
            
            if not comp_info.empty:
                comp_type = comp_info['type'].iloc[0].lower() if 'type' in comp_info.columns else 'unknown'
                
                # Adjust causes based on component type
                if comp_type == 'transformer' and scenario_type in ['high_temperature', 'compound']:
                    # Higher chance of overheating for transformers
                    cause = np.random.choice(['overheating', 'thermal_stress', 'equipment_failure'], p=[0.6, 0.3, 0.1])
                elif comp_type == 'line' and scenario_type in ['high_wind', 'precipitation', 'compound']:
                    # Higher chance of physical damage for lines
                    cause = np.random.choice(['physical_damage', 'wind_damage', 'equipment_failure'], p=[0.5, 0.4, 0.1])
                else:
                    # General case
                    cause = np.random.choice(causes, p=probabilities)
            else:
                # Default if component not found
                cause = np.random.choice(causes, p=probabilities)
                
            failure_causes[comp_id] = cause
            
        return failure_causes
    
    def _create_scenario_metadata(self, generation_method, condition_type):
        """
        Create metadata for a generated scenario.
        
        Args:
            generation_method (str): Method used to generate the scenario.
            condition_type (str): Type of condition (normal, extreme, compound).
            
        Returns:
            dict: Scenario metadata.
        """
        return {
            'generation_method': generation_method,
            'condition_type': condition_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'confidence_score': np.random.uniform(0.7, 0.95)  # Random confidence score for demonstration
        }
    
    def generate_scenarios(self, input_data, count):
        """
        Generate scenarios (to be implemented by subclasses).
        
        Args:
            input_data (dict): Dictionary containing input data.
            count (int): Number of scenarios to generate.
            
        Returns:
            list or dict: Generated scenarios.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_scenarios method")
