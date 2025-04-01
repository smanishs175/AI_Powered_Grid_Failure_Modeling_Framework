#!/usr/bin/env python
"""
Scenario Visualization Example

This script demonstrates how to use the visualization tools to explore 
generated scenarios from the Scenario Generation Module.
"""

import os
import sys
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from gfmf.scenario_generation import ScenarioGenerationModule
from gfmf.scenario_generation.utils.visualization import (
    plot_scenario_distribution,
    plot_failure_heatmap,
    plot_weather_conditions,
    visualize_cascade_network,
    create_scenario_dashboard
)
from gfmf.scenario_generation.utils.export import (
    export_scenarios_to_json,
    export_scenarios_to_csv,
    prepare_for_vulnerability_analysis
)
from gfmf.scenario_generation.utils.data_loader import FailurePredictionDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/scenario_visualization.log')
    ]
)

logger = logging.getLogger('ScenarioVisualizationExample')

def main():
    """Run the scenario visualization example"""
    # Create output directories
    os.makedirs('outputs/scenario_visualizations', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/vulnerability_analysis/input', exist_ok=True)
    
    logger.info("Starting Scenario Visualization Example")
    
    # Step 1: Initialize and run the Scenario Generation Module
    logger.info("Initializing Scenario Generation Module")
    scenario_module = ScenarioGenerationModule()
    
    # Option 1: Generate new scenarios
    # logger.info("Generating new scenarios using synthetic data")
    # scenarios = scenario_module.generate_scenarios(use_synthetic=True)
    
    # Option 2: Load previously generated scenarios if available
    try:
        logger.info("Trying to load previously generated scenarios")
        scenario_path = 'data/scenario_generation/generated_scenarios/all_scenarios.pickle'
        
        with open(scenario_path, 'rb') as f:
            scenarios = pickle.load(f)
        
        logger.info(f"Loaded scenarios from {scenario_path}")
    except (FileNotFoundError, pickle.UnpicklingError):
        logger.info("No previously generated scenarios found, generating new ones")
        scenarios = scenario_module.generate_scenarios(use_synthetic=True)
    
    # Step 2: Create basic visualizations
    logger.info("Creating basic visualizations")
    
    # Plot scenario distribution
    dist_path = plot_scenario_distribution(scenarios, 'outputs/scenario_visualizations')
    logger.info(f"Created scenario distribution plot: {dist_path}")
    
    # Plot component failure heatmap
    heatmap_path = plot_failure_heatmap(scenarios, output_dir='outputs/scenario_visualizations')
    logger.info(f"Created failure heatmap: {heatmap_path}")
    
    # Plot weather conditions
    weather_path = plot_weather_conditions(scenarios, output_dir='outputs/scenario_visualizations')
    logger.info(f"Created weather conditions plot: {weather_path}")
    
    # Step 3: Visualize a cascade network for a specific scenario
    logger.info("Creating cascade network visualization")
    
    # Find a scenario with cascades
    cascade_scenario = None
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
            for scenario in scenario_data:
                if isinstance(scenario, dict) and scenario.get('severity', '') == 'high' and scenario.get('cascade_propagation'):
                    cascade_scenario = scenario
                    break
            if cascade_scenario:
                break
    
    if cascade_scenario:
        cascade_path = visualize_cascade_network(
            cascade_scenario, 
            'outputs/scenario_visualizations',
            f'cascade_network_{cascade_scenario.get("scenario_id", "unknown")}.png'
        )
        logger.info(f"Created cascade network visualization: {cascade_path}")
    else:
        logger.warning("No suitable cascade scenario found for visualization")
    
    # Step 4: Create comprehensive dashboard
    logger.info("Creating comprehensive dashboard")
    
    # Load grid component data directly using pandas
    components_path = 'data/synthetic/synthetic_20250328_144932/synthetic_grid.csv'
    components_df = pd.read_csv(components_path)
    logger.info(f"Loaded {len(components_df)} components from {components_path}")
    
    dashboard_files = create_scenario_dashboard(
        scenarios, 
        components_df,
        'outputs/scenario_visualizations'
    )
    
    logger.info(f"Created dashboard with {len(dashboard_files)} visualizations")
    
    # Step 5: Export for vulnerability analysis
    logger.info("Exporting data for vulnerability analysis")
    
    export_paths = prepare_for_vulnerability_analysis(
        scenarios, 
        components_df,
        'data/vulnerability_analysis/input'
    )
    
    logger.info(f"Exported data to {len(export_paths)} files")
    for key, path in export_paths.items():
        if isinstance(path, list):
            logger.info(f"  {key}: {len(path)} files")
        else:
            logger.info(f"  {key}: {path}")
    
    logger.info("Scenario Visualization Example completed!")
    logger.info("Output files are available in:")
    logger.info("  - outputs/scenario_visualizations/")
    logger.info("  - data/vulnerability_analysis/input/")

if __name__ == "__main__":
    main()
