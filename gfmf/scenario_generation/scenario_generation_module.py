#!/usr/bin/env python
"""
Scenario Generation Module - Main Class

This module handles the generation of realistic grid failure scenarios based on
data from the Failure Prediction Module. It creates normal operating condition scenarios,
extreme weather event scenarios, and compound extreme event scenarios.

It also models cascading failures and validates all scenarios against historical data.
"""

import os
import json
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Import scenario generation components
from .models.scenario_generator import (
    NormalScenarioGenerator,
    ExtremeEventScenarioGenerator,
    CompoundScenarioGenerator
)
from .models.cascading_failure_model import CascadingFailureModel
from .models.scenario_validator import ScenarioValidator
from .utils.data_loader import FailurePredictionDataLoader
from .utils.model_utils import load_config, setup_logger

class ScenarioGenerationModule:
    """
    Main class for the Scenario Generation Module.
    
    This class orchestrates the generation of realistic grid failure scenarios,
    cascade modeling, and scenario validation.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Scenario Generation Module.
        
        Args:
            config_path (str, optional): Path to the configuration file. If None,
                uses the default configuration.
        """
        # Setup logger
        self.logger = setup_logger('ScenarioGenerationModule')
        self.logger.info("Initializing Scenario Generation Module")
        
        # Load configuration
        default_config_path = os.path.join(
            os.path.dirname(__file__), 
            'config', 
            'default_config.yaml'
        )
        self.config = load_config(config_path or default_config_path)
        self.logger.info(f"Loaded configuration from {config_path or default_config_path}")
        
        # Initialize data paths
        self.output_base_path = self.config.get('output_paths', {}).get(
            'base_path', 'data/scenario_generation/'
        )
        
        # Create output directories if they don't exist
        Path(os.path.join(self.output_base_path, 'generated_scenarios')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_base_path, 'cascade_models')).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.normal_generator = NormalScenarioGenerator(self.config.get('normal_scenarios', {}))
        self.extreme_generator = ExtremeEventScenarioGenerator(self.config.get('extreme_scenarios', {}))
        self.compound_generator = CompoundScenarioGenerator(self.config.get('compound_scenarios', {}))
        self.cascading_failure_model = CascadingFailureModel(self.config.get('cascade_model', {}))
        self.scenario_validator = ScenarioValidator(self.config.get('validation', {}))
        self.data_loader = FailurePredictionDataLoader(self.config.get('data_paths', {}))

    def generate_scenarios(self, use_synthetic=False, input_data=None):
        """
        Generate grid failure scenarios based on failure prediction models.
        
        Args:
            use_synthetic (bool): If True, use synthetic data instead of loading from 
                                  failure prediction outputs.
            input_data (dict, optional): Dictionary containing input data. If None,
                                         data is loaded from failure prediction outputs.
        
        Returns:
            dict: Dictionary containing all generated scenarios and validation results.
        """
        self.logger.info("Starting scenario generation process")
        
        # Load input data from failure prediction module
        if input_data is None:
            self.logger.info("Loading input data from failure prediction module")
            input_data = self.data_loader.load_data(use_synthetic=use_synthetic)
        
        # Validate input data
        self._validate_input_data(input_data)
        
        # Generate normal scenarios
        self.logger.info("Generating normal operating condition scenarios")
        normal_scenarios = self.normal_generator.generate_scenarios(
            input_data, 
            count=self.config.get('normal_scenarios', {}).get('count', 50)
        )
        self.logger.info(f"Generated {len(normal_scenarios)} normal scenarios")
        
        # Generate extreme event scenarios
        self.logger.info("Generating extreme event scenarios")
        extreme_scenarios = self.extreme_generator.generate_scenarios(
            input_data,
            event_types=self.config.get('extreme_scenarios', {}).get('event_types', []),
            count_per_type=self.config.get('extreme_scenarios', {}).get('count_per_type', 20)
        )
        
        # Log extreme scenario counts by type
        for event_type, scenarios in extreme_scenarios.items():
            if event_type != 'compound':
                self.logger.info(f"Generated {len(scenarios)} {event_type} scenarios")
        
        # Generate compound scenarios
        self.logger.info("Generating compound extreme event scenarios")
        compound_scenarios = self.compound_generator.generate_scenarios(
            input_data,
            extreme_scenarios=extreme_scenarios,
            count=self.config.get('compound_scenarios', {}).get('count', 30)
        )
        extreme_scenarios['compound'] = compound_scenarios
        self.logger.info(f"Generated {len(compound_scenarios)} compound scenarios")
        
        # Combine all scenarios for cascade modeling
        all_scenarios = {
            'normal': normal_scenarios,
            **extreme_scenarios
        }
        
        # Save generated scenarios
        self._save_scenarios(normal_scenarios, extreme_scenarios)
        
        # Model cascading failures for all scenarios
        self.logger.info("Modeling cascading failures")
        cascade_results = self.cascading_failure_model.model_cascading_failures(
            input_data,
            all_scenarios
        )
        
        # Validate scenarios
        self.logger.info("Validating scenarios")
        validation_results = self.scenario_validator.validate_scenarios(
            all_scenarios,
            historical_data=input_data.get('outage_records', None)
        )
        
        # Compile all results
        generation_results = {
            'normal_scenarios': normal_scenarios,
            'extreme_scenarios': extreme_scenarios,
            'cascade_results': cascade_results,
            'validation_results': validation_results
        }
        
        # Save compiled results
        self._save_compiled_results(generation_results)
        
        return generation_results
    
    def _validate_input_data(self, input_data):
        """
        Validate input data from the failure prediction module.
        
        Args:
            input_data (dict): Dictionary containing input data.
            
        Raises:
            ValueError: If required data is missing or invalid.
        """
        required_keys = [
            'components', 
            'failure_probabilities',
            'environmental_models',
            'extreme_event_models'
        ]
        
        missing_keys = [key for key in required_keys if key not in input_data]
        
        if missing_keys:
            missing_str = ', '.join(missing_keys)
            self.logger.warning(f"Missing required input data: {missing_str}")
            
            # Handle missing data with defaults if possible
            if 'failure_probabilities' in missing_keys and 'components' in input_data:
                self.logger.info("Generating default failure probabilities from component data")
                components = input_data['components']
                # Generate default probabilities based on component type and age
                input_data['failure_probabilities'] = self._generate_default_probabilities(components)
                missing_keys.remove('failure_probabilities')
            
            if missing_keys:
                raise ValueError(f"Missing required input data: {missing_str}")
        
        # Validate failure probabilities
        probs = input_data.get('failure_probabilities', {})
        if not all(0 <= p <= 1 for p in probs.values()):
            self.logger.warning("Some failure probabilities are outside the valid range [0,1]")
            # Clip probabilities to valid range
            for comp_id, p in probs.items():
                probs[comp_id] = max(0, min(1, p))
            
        self.logger.info("Input data validation complete")

    def _generate_default_probabilities(self, components):
        """
        Generate default failure probabilities based on component data.
        
        Args:
            components (DataFrame): DataFrame containing component information.
            
        Returns:
            dict: Dictionary mapping component IDs to default failure probabilities.
        """
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
        
        return probabilities
    
    def _save_scenarios(self, normal_scenarios, extreme_scenarios):
        """
        Save generated scenarios to disk.
        
        Args:
            normal_scenarios (list): List of normal scenario dictionaries.
            extreme_scenarios (dict): Dictionary mapping event types to scenario lists.
        """
        # Save normal scenarios
        normal_path = os.path.join(
            self.output_base_path, 
            'generated_scenarios', 
            'normal_scenarios.pkl'
        )
        with open(normal_path, 'wb') as f:
            pickle.dump(normal_scenarios, f)
            
        # Save extreme scenarios
        extreme_path = os.path.join(
            self.output_base_path, 
            'generated_scenarios', 
            'extreme_scenarios.pkl'
        )
        with open(extreme_path, 'wb') as f:
            pickle.dump(extreme_scenarios, f)
            
        self.logger.info(f"Saved scenarios to {self.output_base_path}/generated_scenarios/")
    
    def _save_compiled_results(self, results):
        """
        Save compiled results to disk.
        
        Args:
            results (dict): Dictionary containing all generation results.
        """
        # Save cascade models
        cascade_path = os.path.join(
            self.output_base_path, 
            'cascade_models', 
            'propagation_models.pkl'
        )
        with open(cascade_path, 'wb') as f:
            pickle.dump(results.get('cascade_results', {}).get('propagation_models', {}), f)
        
        # Save validation metrics
        metadata = {
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'validation_metrics': results.get('validation_results', {}),
            'scenario_counts': {
                'normal': len(results.get('normal_scenarios', [])),
                'extreme': {
                    event_type: len(scenarios)
                    for event_type, scenarios in results.get('extreme_scenarios', {}).items()
                }
            }
        }
        
        metadata_path = os.path.join(self.output_base_path, 'scenario_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Saved compiled results to {self.output_base_path}")

    @staticmethod
    def load_scenario_generation_outputs(base_path='data/scenario_generation/'):
        """
        Load outputs from Scenario Generation Module.
        
        Args:
            base_path (str): Base path to the scenario generation outputs.
            
        Returns:
            dict: Dictionary containing loaded scenario generation outputs.
        """
        # Load compiled results (if available)
        results = {}
        
        # Load normal scenarios
        normal_path = os.path.join(base_path, 'generated_scenarios/normal_scenarios.pkl')
        try:
            with open(normal_path, 'rb') as f:
                results['normal_scenarios'] = pickle.load(f)
            print(f"Loaded {len(results['normal_scenarios'])} normal scenarios")
        except FileNotFoundError:
            print(f"Normal scenarios not found at {normal_path}")
            
        # Load extreme scenarios
        extreme_path = os.path.join(base_path, 'generated_scenarios/extreme_scenarios.pkl')
        try:
            with open(extreme_path, 'rb') as f:
                results['extreme_scenarios'] = pickle.load(f)
            print(f"Loaded extreme scenarios for {len(results['extreme_scenarios'])} event types")
        except FileNotFoundError:
            print(f"Extreme scenarios not found at {extreme_path}")
            
        # Load cascade models
        cascade_path = os.path.join(base_path, 'cascade_models/propagation_models.pkl')
        try:
            with open(cascade_path, 'rb') as f:
                results['cascade_models'] = pickle.load(f)
            print(f"Loaded cascade models for {len(results['cascade_models'])} scenario types")
        except FileNotFoundError:
            print(f"Cascade models not found at {cascade_path}")
            
        # Load validation metrics
        validation_path = os.path.join(base_path, 'scenario_metadata.json')
        try:
            with open(validation_path, 'r') as f:
                metadata = json.load(f)
                results['validation_metrics'] = metadata.get('validation_metrics', {})
            print(f"Loaded scenario validation metrics")
        except FileNotFoundError:
            print(f"Scenario metadata not found at {validation_path}")
            
        return results
