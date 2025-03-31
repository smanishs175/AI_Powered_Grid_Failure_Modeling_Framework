#!/usr/bin/env python3
"""
Grid Failure Modeling Framework (GFMF) - Visualization Demo

This script demonstrates the Visualization and Reporting Module (Module 6)
using mock data to simulate the outputs from previous modules.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization_demo.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Visualization-Demo")

# Create output directory for results
output_dir = "outputs/visualization_demo"
os.makedirs(output_dir, exist_ok=True)

# Import the Visualization and Reporting Module
try:
    from gfmf.visualization_reporting.visualization_reporting_module import VisualizationReportingModule
    from gfmf.visualization_reporting.grid_visualization import GridVisualization
    from gfmf.visualization_reporting.performance_visualization import PerformanceVisualization
    from gfmf.visualization_reporting.report_generator import ReportGenerator
    
    logger.info("Successfully imported Visualization and Reporting Module")
except ImportError as e:
    logger.error(f"Error importing Visualization module: {e}")
    sys.exit(1)


def main():
    """Run the visualization demonstration."""
    logger.info("Starting GFMF Visualization Demo")
    logger.info(f"Results will be saved to: {output_dir}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #-------------------------------------------------------------------------
    # 1. Generate Mock Data (simulating outputs from Modules 1-5)
    #-------------------------------------------------------------------------
    
    # Generate mock grid vulnerability data (from Modules 1-2)
    logger.info("Generating mock grid and vulnerability data")
    grid_data = generate_mock_grid_data()
    vulnerability_data = generate_mock_vulnerability_data(grid_data)
    
    # Generate mock model performance data (from Module 3)
    logger.info("Generating mock prediction model performance data")
    prediction_performance = generate_mock_prediction_performance()
    
    # Generate mock scenario impact data (from Module 4)
    logger.info("Generating mock scenario impact data")
    scenario_impacts = generate_mock_scenario_impacts()
    
    # Generate mock agent performance data (from Module 5)
    logger.info("Generating mock RL agent performance data")
    agent_performance = generate_mock_agent_performance()
    
    #-------------------------------------------------------------------------
    # 2. Demonstrate Visualization and Reporting Module
    #-------------------------------------------------------------------------
    
    logger.info("Initializing Visualization and Reporting Module")
    visualization_module = VisualizationReportingModule()
    
    # A. Create Grid Vulnerability Visualizations
    logger.info("Creating grid vulnerability visualizations")
    
    # Network visualization
    network_viz = visualization_module.create_vulnerability_map(
        map_type="network",
        include_weather=True,
        show_predictions=True,
        output_format="png"
    )
    logger.info(f"Created network visualization: {network_viz['file_path'] if isinstance(network_viz, dict) and 'file_path' in network_viz else 'N/A'}")
    
    # Heatmap visualization
    heatmap_viz = visualization_module.create_vulnerability_map(
        map_type="heatmap",
        include_weather=True,
        show_predictions=True,
        output_format="png"
    )
    logger.info(f"Created heatmap visualization: {heatmap_viz['file_path'] if isinstance(heatmap_viz, dict) and 'file_path' in heatmap_viz else 'N/A'}")
    
    # Geographic visualization
    geo_viz = visualization_module.create_vulnerability_map(
        map_type="geographic",
        include_weather=True,
        show_predictions=True,
        output_format="png"
    )
    logger.info(f"Created geographic visualization: {geo_viz['file_path'] if isinstance(geo_viz, dict) and 'file_path' in geo_viz else 'N/A'}")
    
    # B. Create Performance Visualizations
    logger.info("Creating performance visualizations")
    
    # Model performance visualizations
    performance_viz = visualization_module.create_performance_visualizations(
        include_models=["failure_prediction", "rl_agents"],
        metrics=["accuracy", "reward", "outage_reduction"],
        comparison_type="bar_chart",
        output_format="png"
    )
    
    for viz_type, viz_data in performance_viz.items():
        if isinstance(viz_data, dict):
            for name, data in viz_data.items():
                if 'file_path' in data:
                    logger.info(f"Created {viz_type} - {name}: {data['file_path']}")
    
    # C. Generate Reports
    logger.info("Generating reports")
    
    # Daily summary report
    daily_report = visualization_module.generate_report(
        report_type="daily_summary",
        include_sections=["summary", "vulnerability", "predictions", "policies"],
        output_format="html",
        output_path=os.path.join(output_dir, f"daily_summary_{timestamp}.html")
    )
    logger.info(f"Generated daily summary report: {daily_report['file_path'] if isinstance(daily_report, dict) and 'file_path' in daily_report else 'N/A'}")
    
    # Vulnerability assessment report
    vuln_report = visualization_module.generate_report(
        report_type="vulnerability_assessment",
        include_sections=["overview", "component_risks", "weather_impact"],
        output_format="html",
        output_path=os.path.join(output_dir, f"vulnerability_assessment_{timestamp}.html")
    )
    logger.info(f"Generated vulnerability assessment report: {vuln_report['file_path'] if isinstance(vuln_report, dict) and 'file_path' in vuln_report else 'N/A'}")
    
    # Policy evaluation report
    policy_report = visualization_module.generate_report(
        report_type="policy_evaluation",
        include_sections=["overview", "agent_performance", "scenario_evaluations", "recommendations"],
        output_format="html",
        output_path=os.path.join(output_dir, f"policy_evaluation_{timestamp}.html")
    )
    logger.info(f"Generated policy evaluation report: {policy_report['file_path'] if isinstance(policy_report, dict) and 'file_path' in policy_report else 'N/A'}")
    
    # D. Dashboard (commented out as it would block the script)
    # logger.info("Launching dashboard")
    # dashboard = visualization_module.launch_dashboard(
    #     dashboard_type="operational",
    #     auto_refresh=True,
    #     refresh_interval=300
    # )
    # logger.info(f"Launched dashboard at: {dashboard['url'] if isinstance(dashboard, dict) and 'url' in dashboard else 'N/A'}")
    
    logger.info("GFMF Visualization Demo completed successfully")
    logger.info(f"All results saved to: {output_dir}")
    print(f"\nGFMF Visualization Demo completed. All visualizations and reports saved to: {output_dir}\n")


def generate_mock_grid_data():
    """Generate mock grid data."""
    # Create simple grid structure with transformers and transmission lines
    grid_data = {
        "components": [
            {"id": f"transformer_{i}", "type": "transformer", "capacity": 100.0, 
             "location": {"lat": 40 + np.random.rand(), "lon": -100 - np.random.rand()}}
            for i in range(10)
        ] + [
            {"id": f"line_{i}", "type": "transmission_line", "capacity": 80.0, 
             "source": f"transformer_{i//2}", "target": f"transformer_{i//2+1}"}
            for i in range(0, 18, 2)
        ],
        "metadata": {
            "grid_id": "demo_grid",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    return grid_data


def generate_mock_vulnerability_data(grid_data):
    """Generate mock vulnerability data for grid components."""
    # Create vulnerability scores for each component
    vulnerability_data = {
        "component_scores": {},
        "environmental_threats": {
            "extreme_temperature": {
                "weight": 0.3,
                "threshold": 30.0
            },
            "heavy_precipitation": {
                "weight": 0.25,
                "threshold": 15.0
            },
            "high_winds": {
                "weight": 0.45,
                "threshold": 20.0
            }
        },
        "metadata": {
            "model_version": "1.0",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Add vulnerability scores for each component
    for component in grid_data["components"]:
        component_id = component["id"]
        vulnerability_data["component_scores"][component_id] = {
            "score": round(0.2 + 0.6 * np.random.rand(), 2),
            "factors": {
                "age": round(0.1 + 0.4 * np.random.rand(), 2),
                "weather_exposure": round(0.1 + 0.4 * np.random.rand(), 2),
                "load": round(0.1 + 0.4 * np.random.rand(), 2)
            }
        }
    
    return vulnerability_data


def generate_mock_prediction_performance():
    """Generate mock performance metrics for failure prediction models."""
    # Create performance metrics for different prediction models
    prediction_performance = {
        "random_forest": {
            "accuracy": 0.82,
            "precision": 0.76,
            "recall": 0.79,
            "f1_score": 0.77,
            "roc_auc": 0.86,
            "confusion_matrix": {
                "true_positive": 156,
                "false_positive": 49,
                "true_negative": 423,
                "false_negative": 41
            }
        },
        "gradient_boosting": {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
            "roc_auc": 0.89,
            "confusion_matrix": {
                "true_positive": 147,
                "false_positive": 37,
                "true_negative": 435,
                "false_negative": 49
            }
        },
        "neural_network": {
            "accuracy": 0.87,
            "precision": 0.83,
            "recall": 0.81,
            "f1_score": 0.82,
            "roc_auc": 0.91,
            "confusion_matrix": {
                "true_positive": 159,
                "false_positive": 32,
                "true_negative": 440,
                "false_negative": 37
            }
        }
    }
    
    return prediction_performance


def generate_mock_scenario_impacts():
    """Generate mock scenario impact data."""
    # Create impact metrics for different scenario types
    scenario_impacts = {
        "normal_scenarios": [
            {
                "scenario_id": "normal_1",
                "outage_rate": 0.05,
                "components_affected": 2,
                "load_shed_percentage": 0.02,
                "stability_duration": 0.98
            },
            {
                "scenario_id": "normal_2",
                "outage_rate": 0.03,
                "components_affected": 1,
                "load_shed_percentage": 0.01,
                "stability_duration": 0.99
            }
        ],
        "extreme_scenarios": [
            {
                "scenario_id": "extreme_1",
                "outage_rate": 0.25,
                "components_affected": 8,
                "load_shed_percentage": 0.18,
                "stability_duration": 0.72
            },
            {
                "scenario_id": "extreme_2",
                "outage_rate": 0.32,
                "components_affected": 12,
                "load_shed_percentage": 0.27,
                "stability_duration": 0.65
            }
        ],
        "compound_scenarios": [
            {
                "scenario_id": "compound_1",
                "outage_rate": 0.45,
                "components_affected": 16,
                "load_shed_percentage": 0.38,
                "stability_duration": 0.55
            },
            {
                "scenario_id": "compound_2",
                "outage_rate": 0.52,
                "components_affected": 19,
                "load_shed_percentage": 0.45,
                "stability_duration": 0.42
            }
        ]
    }
    
    return scenario_impacts


def generate_mock_agent_performance():
    """Generate mock performance metrics for reinforcement learning agents."""
    # Create agent performance metrics for different algorithms
    agent_types = ["dqn", "ppo", "sac", "td3", "gail"]
    scenario_types = ["normal", "extreme", "compound"]
    
    agent_performance = {
        agent: {
            "average_reward": {
                scenario: round(200 + 300 * np.random.rand(), 2) for scenario in scenario_types
            },
            "outage_reduction": {
                scenario: round(0.3 + 0.6 * np.random.rand(), 2) for scenario in scenario_types
            },
            "stability_duration": {
                scenario: round(0.5 + 0.5 * np.random.rand(), 2) for scenario in scenario_types
            },
            "learning_curve": {
                "steps": list(range(0, 110000, 10000)),
                "rewards": [round(-500 + i/1000 + 50*np.random.rand(), 2) for i in range(0, 110000, 10000)]
            }
        } for agent in agent_types
    }
    
    return agent_performance


if __name__ == "__main__":
    main()
