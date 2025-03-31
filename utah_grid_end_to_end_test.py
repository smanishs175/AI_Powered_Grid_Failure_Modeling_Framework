#!/usr/bin/env python3
"""
Utah Grid End-to-End Test for Grid Failure Modeling Framework

This script runs the complete Grid Failure Modeling Framework (GFMF) pipeline from
Module 1 through Module 6 using synthetic Utah grid data.

Steps:
1. Generate Utah test data
2. Run the Data Management Module (Module 1)
3. Run the Vulnerability Analysis Module (Module 2)
4. Run the Failure Prediction Module (Module 3)
5. Run the Scenario Generation Module (Module 4)
6. Run the Reinforcement Learning Module (Module 5)
7. Run the Visualization and Reporting Module (Module 6)
"""

import os
import sys
import logging
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import the Utah grid data generator
sys.path.append('test_data/utah_grid')
from utah_grid_generator import save_test_data

# Import GFMF modules
from gfmf.data_management.data_management_module import DataManagementModule
from gfmf.vulnerability_analysis.vulnerability_analysis_module import VulnerabilityAnalysisModule
from gfmf.failure_prediction.failure_prediction_module import FailurePredictionModule
from gfmf.scenario_generation.scenario_generation_module import ScenarioGenerationModule
from gfmf.reinforcement_learning.reinforcement_learning_module import ReinforcementLearningModule
from gfmf.visualization_reporting.visualization_reporting_module import VisualizationReportingModule

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('utah_grid_test_run.log')
    ]
)

logger = logging.getLogger('Utah-Grid-E2E-Test')

# Directories
DATA_DIR = 'test_data/utah_grid'
OUTPUT_DIR = 'outputs/utah_grid_test'
CONFIG_DIR = 'config'

# Create directory structure
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_data_management_module():
    """Run the Data Management Module (Module 1)."""
    logger.info("Running Data Management Module")
    
    # Initialize the module
    data_module = DataManagementModule()
    
    # Load the Utah grid test data
    grid_topology_path = os.path.join(DATA_DIR, 'utah_grid_topology.json')
    weather_data_path = os.path.join(DATA_DIR, 'utah_weather_data.csv')
    outage_data_path = os.path.join(DATA_DIR, 'utah_outage_data.csv')
    
    with open(grid_topology_path, 'r') as f:
        grid_topology = json.load(f)
    
    components_df = pd.DataFrame(grid_topology['components'])
    connections_df = pd.DataFrame(grid_topology['connections'])
    weather_df = pd.read_csv(weather_data_path)
    outage_df = pd.read_csv(outage_data_path)
    
    # Process the data
    processed_data = data_module.process_data(
        grid_components=components_df,
        grid_connections=connections_df,
        weather_data=weather_df,
        outage_data=outage_df
    )
    
    # Feature engineering
    features_df = data_module.engineer_features(processed_data)
    
    # Save processed data
    output_path = os.path.join(OUTPUT_DIR, 'module_1_processed_data')
    os.makedirs(output_path, exist_ok=True)
    
    features_df.to_csv(os.path.join(output_path, 'processed_features.csv'), index=False)
    
    logger.info(f"Data Management Module completed. Processed data saved to {output_path}")
    
    return features_df


def run_vulnerability_analysis_module(features_df):
    """Run the Vulnerability Analysis Module (Module 2)."""
    logger.info("Running Vulnerability Analysis Module")
    
    # Initialize the module
    vulnerability_module = VulnerabilityAnalysisModule()
    
    # Analyze component vulnerabilities
    component_vulnerabilities = vulnerability_module.analyze_component_vulnerabilities(features_df)
    
    # Analyze environmental threats
    environmental_threats = vulnerability_module.analyze_environmental_threats(features_df)
    
    # Calculate overall vulnerability scores
    vulnerability_scores = vulnerability_module.calculate_vulnerability_scores(
        component_vulnerabilities, 
        environmental_threats
    )
    
    # Save vulnerability analysis results
    output_path = os.path.join(OUTPUT_DIR, 'module_2_vulnerability_analysis')
    os.makedirs(output_path, exist_ok=True)
    
    vulnerability_scores.to_csv(os.path.join(output_path, 'vulnerability_scores.csv'), index=False)
    
    logger.info(f"Vulnerability Analysis Module completed. Results saved to {output_path}")
    
    return vulnerability_scores


def run_failure_prediction_module(features_df, vulnerability_scores):
    """Run the Failure Prediction Module (Module 3)."""
    logger.info("Running Failure Prediction Module")
    
    # Initialize the module
    prediction_module = FailurePredictionModule()
    
    # Prepare data for prediction
    X, y = prediction_module.prepare_data(features_df, vulnerability_scores)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = prediction_module.split_data(X, y)
    
    # Train the prediction model
    model = prediction_module.train_model(X_train, y_train)
    
    # Evaluate the model
    evaluation_metrics = prediction_module.evaluate_model(model, X_test, y_test)
    
    # Predict failures
    failure_predictions = prediction_module.predict_failures(model, X_test)
    
    # Save prediction results
    output_path = os.path.join(OUTPUT_DIR, 'module_3_failure_prediction')
    os.makedirs(output_path, exist_ok=True)
    
    pd.DataFrame(evaluation_metrics, index=[0]).to_csv(
        os.path.join(output_path, 'evaluation_metrics.csv'), index=False
    )
    
    failure_predictions.to_csv(os.path.join(output_path, 'failure_predictions.csv'), index=False)
    
    logger.info(f"Failure Prediction Module completed. Results saved to {output_path}")
    
    return failure_predictions, model


def run_scenario_generation_module(vulnerability_scores, failure_predictions):
    """Run the Scenario Generation Module (Module 4)."""
    logger.info("Running Scenario Generation Module")
    
    # Initialize the module
    scenario_module = ScenarioGenerationModule()
    
    # Generate baseline scenario
    baseline_scenario = scenario_module.generate_baseline_scenario(vulnerability_scores)
    
    # Generate failure scenarios
    failure_scenarios = scenario_module.generate_failure_scenarios(
        vulnerability_scores, 
        failure_predictions
    )
    
    # Generate extreme weather scenarios
    weather_scenarios = scenario_module.generate_weather_scenarios(vulnerability_scores)
    
    # Calculate impacts
    scenario_impacts = scenario_module.calculate_scenario_impacts(
        [baseline_scenario] + failure_scenarios + weather_scenarios
    )
    
    # Save scenario results
    output_path = os.path.join(OUTPUT_DIR, 'module_4_scenario_generation')
    os.makedirs(output_path, exist_ok=True)
    
    scenario_impacts.to_csv(os.path.join(output_path, 'scenario_impacts.csv'), index=False)
    
    logger.info(f"Scenario Generation Module completed. Results saved to {output_path}")
    
    return scenario_impacts


def run_reinforcement_learning_module(vulnerability_scores, scenario_impacts):
    """Run the Reinforcement Learning Module (Module 5)."""
    logger.info("Running Reinforcement Learning Module")
    
    # Initialize the module
    rl_module = ReinforcementLearningModule()
    
    # Create environment
    env = rl_module.create_environment(vulnerability_scores, scenario_impacts)
    
    # Train SAC agent
    sac_agent = rl_module.train_agent(env, agent_type='sac', total_timesteps=10000)
    
    # Train TD3 agent
    td3_agent = rl_module.train_agent(env, agent_type='td3', total_timesteps=10000)
    
    # Evaluate agents
    sac_performance = rl_module.evaluate_agent(sac_agent, env)
    td3_performance = rl_module.evaluate_agent(td3_agent, env)
    
    # Extract policy
    best_agent = sac_agent if sac_performance['mean_reward'] > td3_performance['mean_reward'] else td3_agent
    best_agent_type = 'sac' if sac_performance['mean_reward'] > td3_performance['mean_reward'] else 'td3'
    
    hardening_policy = rl_module.extract_policy(best_agent, vulnerability_scores)
    
    # Save RL results
    output_path = os.path.join(OUTPUT_DIR, 'module_5_reinforcement_learning')
    os.makedirs(output_path, exist_ok=True)
    
    # Save performance metrics
    performance_metrics = {
        'sac': sac_performance,
        'td3': td3_performance
    }
    
    with open(os.path.join(output_path, 'agent_performance.json'), 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    # Save policy
    hardening_policy.to_csv(os.path.join(output_path, 'hardening_policy.csv'), index=False)
    
    # Save agent models
    sac_agent.save(os.path.join(output_path, 'sac_agent'))
    td3_agent.save(os.path.join(output_path, 'td3_agent'))
    
    logger.info(f"Reinforcement Learning Module completed. Results saved to {output_path}")
    
    return hardening_policy, performance_metrics, best_agent_type


def run_visualization_reporting_module(
    grid_topology, 
    vulnerability_scores, 
    failure_predictions, 
    scenario_impacts, 
    hardening_policy, 
    agent_performance
):
    """Run the Visualization and Reporting Module (Module 6)."""
    logger.info("Running Visualization and Reporting Module")
    
    # Initialize the module
    viz_module = VisualizationReportingModule()
    
    # Output directory
    viz_output_dir = os.path.join(OUTPUT_DIR, 'module_6_visualization_reporting')
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Create vulnerability maps
    network_viz = viz_module.create_vulnerability_map(
        map_type='network',
        include_weather=True,
        show_predictions=True,
        output_format='png'
    )
    
    heatmap_viz = viz_module.create_vulnerability_map(
        map_type='heatmap',
        include_weather=True,
        show_predictions=True,
        output_format='png'
    )
    
    geographic_viz = viz_module.create_vulnerability_map(
        map_type='geographic',
        include_weather=True,
        show_predictions=True,
        output_format='png'
    )
    
    # Create performance visualizations
    performance_viz = viz_module.create_performance_visualizations(
        include_models=["failure_prediction", "rl_agents"],
        metrics=["accuracy", "reward", "outage_reduction"],
        comparison_type="bar_chart",
        output_format="png"
    )
    
    # Generate reports
    summary_report = viz_module.generate_report(
        report_type='daily_summary',
        include_sections=['overview', 'vulnerabilities', 'predictions', 'recommendations'],
        output_format='html',
        output_path=os.path.join(viz_output_dir, 'utah_grid_summary_report.html')
    )
    
    vulnerability_report = viz_module.generate_report(
        report_type='vulnerability_assessment',
        include_sections=['component_vulnerabilities', 'environmental_threats', 'risk_zones'],
        output_format='html',
        output_path=os.path.join(viz_output_dir, 'utah_grid_vulnerability_report.html')
    )
    
    policy_report = viz_module.generate_report(
        report_type='policy_evaluation',
        include_sections=['policy_details', 'performance_metrics', 'cost_benefit'],
        output_format='html',
        output_path=os.path.join(viz_output_dir, 'utah_grid_policy_report.html')
    )
    
    # Copy and rename the visualization files for the reports directory
    for viz_type, viz_data in network_viz.items():
        src = viz_data.get('file_path')
        if src and os.path.exists(src):
            dest = os.path.join(viz_output_dir, f'utah_network_viz_{viz_type}.png')
            os.system(f'cp "{src}" "{dest}"')
    
    for viz_type, viz_data in performance_viz.items():
        src = viz_data.get('file_path')
        if src and os.path.exists(src):
            dest = os.path.join(viz_output_dir, f'utah_performance_{viz_type}.png')
            os.system(f'cp "{src}" "{dest}"')
    
    logger.info(f"Visualization and Reporting Module completed. Results saved to {viz_output_dir}")
    
    return {
        'network_viz': network_viz,
        'heatmap_viz': heatmap_viz,
        'geographic_viz': geographic_viz,
        'performance_viz': performance_viz,
        'summary_report': summary_report,
        'vulnerability_report': vulnerability_report,
        'policy_report': policy_report
    }


def run_end_to_end_test():
    """Run the complete Grid Failure Modeling Framework pipeline."""
    logger.info("Starting Utah Grid End-to-End Test for Grid Failure Modeling Framework")
    logger.info(f"Results will be saved to: {OUTPUT_DIR}")
    
    start_time = time.time()
    
    # Generate Utah grid test data
    logger.info("Generating Utah grid test data")
    save_test_data(DATA_DIR)
    
    # Load grid topology data for later use
    with open(os.path.join(DATA_DIR, 'utah_grid_topology.json'), 'r') as f:
        grid_topology = json.load(f)
    
    # Run Module 1: Data Management
    features_df = run_data_management_module()
    
    # Run Module 2: Vulnerability Analysis
    vulnerability_scores = run_vulnerability_analysis_module(features_df)
    
    # Run Module 3: Failure Prediction
    failure_predictions, prediction_model = run_failure_prediction_module(features_df, vulnerability_scores)
    
    # Run Module 4: Scenario Generation
    scenario_impacts = run_scenario_generation_module(vulnerability_scores, failure_predictions)
    
    # Run Module 5: Reinforcement Learning
    hardening_policy, agent_performance, best_agent = run_reinforcement_learning_module(
        vulnerability_scores, scenario_impacts
    )
    
    # Run Module 6: Visualization and Reporting
    visualization_results = run_visualization_reporting_module(
        grid_topology,
        vulnerability_scores,
        failure_predictions,
        scenario_impacts,
        hardening_policy,
        agent_performance
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Utah Grid End-to-End Test completed in {total_time:.2f} seconds")
    logger.info(f"All results saved to: {OUTPUT_DIR}")
    
    # Print summary of results
    print("\n" + "="*80)
    print("UTAH GRID FAILURE MODELING FRAMEWORK TEST SUMMARY")
    print("="*80)
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Number of grid components analyzed: {len(grid_topology['components'])}")
    print(f"Number of connections analyzed: {len(grid_topology['connections'])}")
    print(f"Number of vulnerabilities identified: {len(vulnerability_scores)}")
    print(f"Failure prediction model accuracy: {agent_performance.get(best_agent, {}).get('mean_reward', 0):.4f}")
    print(f"Number of scenarios generated: {len(scenario_impacts)}")
    print(f"Best reinforcement learning agent: {best_agent}")
    print(f"Number of hardening policy recommendations: {len(hardening_policy)}")
    print(f"Reports generated: 3 (Summary, Vulnerability Assessment, Policy Evaluation)")
    print("="*80)
    print(f"All results are available in: {OUTPUT_DIR}")
    print(f"Visualization reports available in: {os.path.join(OUTPUT_DIR, 'module_6_visualization_reporting')}")
    print("="*80)


if __name__ == "__main__":
    run_end_to_end_test()
