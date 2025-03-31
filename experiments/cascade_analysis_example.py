#!/usr/bin/env python
"""
Cascade Analysis Example

This script demonstrates the enhanced cascading failure modeling capabilities.
It generates scenarios, runs cascade simulations, and visualizes the results in detail.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import yaml
from typing import Dict, List, Any, Optional

# Ensure gfmf is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gfmf.scenario_generation.scenario_generation_module import ScenarioGenerationModule
from gfmf.scenario_generation.models.cascading_failure_model import CascadingFailureModel
from gfmf.scenario_generation.utils.data_loader import FailurePredictionDataLoader
from gfmf.scenario_generation.utils.visualization import visualize_cascade_network

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CascadeAnalysisExample')

OUTPUT_DIR = 'outputs/cascade_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    """
    Main function to demonstrate cascading failure analysis.
    """
    logger.info("Starting Cascade Analysis Example")
    
    # Load grid component data for cascade modeling
    logger.info("Loading grid component data")
    data_loader = FailurePredictionDataLoader()
    data = data_loader.load_data()
    
    # Create a DataFrame for components if using direct data
    if not isinstance(data['components'], pd.DataFrame):
        components_df = pd.DataFrame(data['components'])
    else:
        components_df = data['components']
    
    # Run cascade analysis on synthetic data
    run_cascade_analysis(components_df)
    
    logger.info("Cascade Analysis Example completed!")
    logger.info(f"Output files are available in: {OUTPUT_DIR}")


def run_cascade_analysis(components_df: pd.DataFrame):
    """
    Run cascade analysis and create visualizations.
    
    Args:
        components_df: DataFrame containing component information
    """
    logger.info("Setting up cascade analysis")
    
    # Create a cascading failure model with default config
    cascade_model = CascadingFailureModel(config={
        'max_cascade_steps': 10,
        'load_redistribution_factor': 0.7,
        'capacity_threshold': 0.85
    })
    
    # Generate several test scenarios with different initial failures
    test_scenarios = generate_test_scenarios(components_df)
    
    # Organize scenarios in the format expected by the model
    scenarios = {
        'normal': test_scenarios['1-component'],
        'high_impact': test_scenarios['3-component'],
        'extreme': test_scenarios['5-component']
    }
    
    # Prepare input data for the model
    input_data = {
        'components': components_df
    }
    
    # Model cascading failures
    logger.info("Modeling cascading failures")
    cascade_results = cascade_model.model_cascading_failures(input_data, scenarios)
    
    # Analyze and visualize results
    analyze_cascade_results(cascade_results, scenarios, output_dir=OUTPUT_DIR)


def generate_test_scenarios(components_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Generate test scenarios with different numbers of initial failures.
    
    Args:
        components_df: DataFrame containing component information
        
    Returns:
        Dictionary of test scenarios grouped by complexity
    """
    logger.info("Generating test scenarios with different initial failures")
    
    # Get component IDs
    component_ids = components_df['component_id'].tolist()
    
    # Generate scenarios with different numbers of initial failures
    scenarios = {
        '1-component': [],
        '3-component': [],
        '5-component': []
    }
    
    # Generate scenarios with 1, 3, and 5 initial component failures
    for i in range(5):  # Generate 5 scenarios of each type
        # Single component failure scenarios (simple)
        single_comp = np.random.choice(component_ids)
        
        # Calculate random failure probability
        failure_prob = np.random.uniform(0.7, 0.9)
        
        single_scenario = {
            'scenario_id': f'single_{i+1}',
            'scenario_type': 'normal',
            'severity': 'low',
            'timestamp': '2025-04-01T12:00:00',
            'component_failures': {
                single_comp: {
                    'probability': failure_prob,
                    'cause': 'equipment_failure'
                }
            },
            'weather_conditions': {
                'temperature': 25.0,
                'wind_speed': 5.0,
                'precipitation': 0.0
            }
        }
        scenarios['1-component'].append(single_scenario)
        
        # 3-component failure scenarios (medium complexity)
        failed_comps = np.random.choice(component_ids, size=3, replace=False)
        three_comp_scenario = {
            'scenario_id': f'three_comp_{i+1}',
            'scenario_type': 'high_impact',
            'severity': 'medium',
            'timestamp': '2025-04-01T12:00:00',
            'component_failures': {}
        }
        
        for comp in failed_comps:
            three_comp_scenario['component_failures'][comp] = {
                'probability': np.random.uniform(0.7, 0.95),
                'cause': np.random.choice(['equipment_failure', 'weather', 'external_impact'])
            }
            
        three_comp_scenario['weather_conditions'] = {
            'temperature': 30.0,
            'wind_speed': 12.0,
            'precipitation': 5.0
        }
        scenarios['3-component'].append(three_comp_scenario)
        
        # 5-component failure scenarios (high complexity)
        failed_comps = np.random.choice(component_ids, size=5, replace=False)
        five_comp_scenario = {
            'scenario_id': f'five_comp_{i+1}',
            'scenario_type': 'extreme',
            'severity': 'high',
            'timestamp': '2025-04-01T12:00:00',
            'component_failures': {}
        }
        
        for comp in failed_comps:
            five_comp_scenario['component_failures'][comp] = {
                'probability': np.random.uniform(0.8, 0.99),
                'cause': np.random.choice(['equipment_failure', 'weather', 'external_impact', 'cyber_attack'])
            }
            
        five_comp_scenario['weather_conditions'] = {
            'temperature': 35.0,
            'wind_speed': 25.0,
            'precipitation': 15.0
        }
        scenarios['5-component'].append(five_comp_scenario)
    
    logger.info(f"Generated {sum(len(s) for s in scenarios.values())} test scenarios")
    return scenarios


def analyze_cascade_results(cascade_results: Dict, 
                          scenarios: Dict[str, List[Dict]], 
                          output_dir: str = OUTPUT_DIR):
    """
    Analyze and visualize cascading failure results.
    
    Args:
        cascade_results: Results from the cascading failure model
        scenarios: Original scenarios
        output_dir: Directory to save visualizations
    """
    logger.info("Analyzing cascading failure results")
    
    # Extract results for different scenario types
    results_by_type = cascade_results['results']
    
    # Create summary statistics
    summary = {
        'scenario_type': [],
        'avg_failures': [],
        'max_failures': [],
        'avg_cascade_steps': [],
        'max_cascade_steps': []
    }
    
    # Track scenarios with most significant cascades for visualization
    best_cascades = {
        'max_failures': None,
        'max_steps': None,
        'max_chain': None
    }
    max_failure_count = 0
    max_step_count = 0
    max_chain_length = 0
    
    # Process each scenario type
    for scenario_type, scenario_results in results_by_type.items():
        failures = []
        steps = []
        
        # Process individual scenarios
        for result in scenario_results:
            scenario_id = result['scenario_id']
            cascade = result['cascade']
            
            # Collect statistics
            failures.append(cascade['final_failed_count'])
            steps.append(cascade['total_steps'])
            
            # Check for significant cascades
            if cascade['final_failed_count'] > max_failure_count:
                max_failure_count = cascade['final_failed_count']
                best_cascades['max_failures'] = result
                
            if cascade['total_steps'] > max_step_count:
                max_step_count = cascade['total_steps']
                best_cascades['max_steps'] = result
            
            # Check for longest failure chain
            if 'failure_chains' in cascade:
                for chain in cascade['failure_chains']:
                    chain_length = chain.get('length', 0)
                    if chain_length > max_chain_length:
                        max_chain_length = chain_length
                        best_cascades['max_chain'] = result
        
        # Add to summary
        summary['scenario_type'].append(scenario_type)
        summary['avg_failures'].append(np.mean(failures) if failures else 0)
        summary['max_failures'].append(np.max(failures) if failures else 0)
        summary['avg_cascade_steps'].append(np.mean(steps) if steps else 0)
        summary['max_cascade_steps'].append(np.max(steps) if steps else 0)
    
    # Convert to DataFrame for display
    summary_df = pd.DataFrame(summary)
    logger.info("Cascade summary statistics:")
    logger.info("\n" + str(summary_df))
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(output_dir, 'cascade_summary.csv'), index=False)
    
    # Create visualizations for the most significant cascades
    for cascade_type, scenario in best_cascades.items():
        if scenario:
            logger.info(f"Visualizing {cascade_type} cascade scenario: {scenario['scenario_id']}")
            
            # Standard visualization
            output_file = f"{cascade_type}_cascade.png"
            visualize_cascade_network(
                scenario,
                output_dir=output_dir,
                filename=output_file,
                show_component_types=True
            )
            
            # Detailed visualization
            output_file_detailed = f"{cascade_type}_cascade_detailed.png"
            visualize_cascade_network(
                scenario,
                output_dir=output_dir,
                filename=output_file_detailed,
                show_component_types=False,
                detailed_labels=True
            )
            
            logger.info(f"Created visualization: {os.path.join(output_dir, output_file)}")
            logger.info(f"Created detailed visualization: {os.path.join(output_dir, output_file_detailed)}")


if __name__ == "__main__":
    main()
