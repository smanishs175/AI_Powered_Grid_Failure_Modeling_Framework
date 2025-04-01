#!/usr/bin/env python3
"""
Grid Failure Modeling Framework (GFMF) Integration Demo

This script demonstrates the complete pipeline from data management to visualization,
integrating all six modules of the GFMF.
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
        logging.FileHandler("gfmf_demo.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GFMF-Demo")

# Create output directory for results
output_dir = "outputs/integration_demo"
os.makedirs(output_dir, exist_ok=True)

# Import modules (with error handling for any missing dependencies)
try:
    # Module 1: Data Management
    from gfmf.data_management.data_management_module import DataManagementModule

    # Module 2: Vulnerability Analysis
    from gfmf.vulnerability_analysis.vulnerability_analysis_module import VulnerabilityAnalysisModule

    # Module 3: Failure Prediction
    from gfmf.failure_prediction.failure_prediction_module import FailurePredictionModule

    # Module 4: Scenario Generation
    from gfmf.scenario_generation.scenario_generation_module import ScenarioGenerationModule

    # Module 5: Reinforcement Learning
    from gfmf.reinforcement_learning.reinforcement_learning_module import ReinforcementLearningModule

    # Module 6: Visualization and Reporting
    from gfmf.visualization_reporting.visualization_reporting_module import VisualizationReportingModule
    
    logger.info("Successfully imported all GFMF modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.info("Running in limited mode with only available modules")


def main():
    """Run the complete GFMF pipeline."""
    logger.info("Starting GFMF Integration Demo")
    logger.info(f"Results will be saved to: {output_dir}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Data Management
    logger.info("Step 1: Data Management Module")
    try:
        data_module = DataManagementModule()
        
        # Load grid data
        grid_data = data_module.load_grid_data()
        logger.info(f"Loaded grid data with {len(grid_data['components'])} components")
        
        # Load weather data
        weather_data = data_module.load_weather_data()
        logger.info(f"Loaded weather data with {len(weather_data.index) if isinstance(weather_data, pd.DataFrame) else 'N/A'} records")
        
        # Load outage data
        outage_data = data_module.load_outage_data()
        logger.info(f"Loaded outage data with {len(outage_data.index) if isinstance(outage_data, pd.DataFrame) else 'N/A'} records")
        
        # Preprocess data
        processed_data = data_module.preprocess_data(grid_data, weather_data, outage_data)
        logger.info("Data preprocessing completed")
        
        # Save processed data
        data_module.save_processed_data(processed_data, os.path.join(output_dir, f"processed_data_{timestamp}.csv"))
        logger.info("Data management step completed successfully")
    except Exception as e:
        logger.error(f"Error in Data Management Module: {e}")
        # Continue with demo using mock data
        logger.info("Continuing with demo using synthetic data")
        processed_data = generate_mock_data()
    
    # Step 2: Vulnerability Analysis
    logger.info("Step 2: Vulnerability Analysis Module")
    try:
        vulnerability_module = VulnerabilityAnalysisModule()
        
        # Profile component vulnerabilities
        vulnerability_scores = vulnerability_module.analyze_component_vulnerabilities(processed_data)
        logger.info(f"Generated vulnerability scores for {len(vulnerability_scores) if isinstance(vulnerability_scores, dict) else 'N/A'} components")
        
        # Analyze environmental threats
        threat_profiles = vulnerability_module.analyze_environmental_threats(processed_data)
        logger.info(f"Generated threat profiles for {len(threat_profiles) if isinstance(threat_profiles, dict) else 'N/A'} threat types")
        
        # Create vulnerability model
        vulnerability_model = vulnerability_module.create_vulnerability_model(vulnerability_scores, threat_profiles)
        logger.info("Created vulnerability model")
        
        # Save vulnerability results
        vulnerability_module.save_vulnerability_analysis(vulnerability_model, os.path.join(output_dir, f"vulnerability_analysis_{timestamp}.json"))
        logger.info("Vulnerability analysis step completed successfully")
    except Exception as e:
        logger.error(f"Error in Vulnerability Analysis Module: {e}")
        vulnerability_model = generate_mock_vulnerability_model()
    
    # Step 3: Failure Prediction
    logger.info("Step 3: Failure Prediction Module")
    try:
        prediction_module = FailurePredictionModule()
        
        # Train prediction model
        prediction_model = prediction_module.train_prediction_model(processed_data, vulnerability_model)
        logger.info("Trained failure prediction model")
        
        # Evaluate model performance
        performance_metrics = prediction_module.evaluate_model(prediction_model)
        logger.info(f"Model evaluation metrics: {performance_metrics}")
        
        # Make predictions
        predictions = prediction_module.predict_failures(prediction_model, processed_data)
        logger.info(f"Generated predictions for {len(predictions) if isinstance(predictions, dict) else 'N/A'} components")
        
        # Save prediction results
        prediction_module.save_predictions(predictions, os.path.join(output_dir, f"failure_predictions_{timestamp}.csv"))
        logger.info("Failure prediction step completed successfully")
    except Exception as e:
        logger.error(f"Error in Failure Prediction Module: {e}")
        predictions = generate_mock_predictions()
        performance_metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.78, "f1": 0.80}
    
    # Step 4: Scenario Generation
    logger.info("Step 4: Scenario Generation Module")
    try:
        scenario_module = ScenarioGenerationModule()
        
        # Generate normal operating scenarios
        normal_scenarios = scenario_module.generate_normal_scenarios(processed_data, vulnerability_model)
        logger.info(f"Generated {len(normal_scenarios) if isinstance(normal_scenarios, list) else 'N/A'} normal operating scenarios")
        
        # Generate extreme scenarios
        extreme_scenarios = scenario_module.generate_extreme_scenarios(processed_data, vulnerability_model)
        logger.info(f"Generated {len(extreme_scenarios) if isinstance(extreme_scenarios, list) else 'N/A'} extreme scenarios")
        
        # Generate combined scenarios
        combined_scenarios = scenario_module.generate_compound_scenarios(processed_data, vulnerability_model)
        logger.info(f"Generated {len(combined_scenarios) if isinstance(combined_scenarios, list) else 'N/A'} compound scenarios")
        
        # Simulate cascade failures
        cascade_results = scenario_module.simulate_cascade_failures(combined_scenarios)
        logger.info("Simulated cascade failures for compound scenarios")
        
        # Save scenario results
        scenario_module.save_scenarios(normal_scenarios + extreme_scenarios + combined_scenarios, 
                                      os.path.join(output_dir, f"generated_scenarios_{timestamp}.json"))
        logger.info("Scenario generation step completed successfully")
    except Exception as e:
        logger.error(f"Error in Scenario Generation Module: {e}")
        normal_scenarios, extreme_scenarios, combined_scenarios = generate_mock_scenarios()
        cascade_results = {"propagation_paths": [], "impact_metrics": []}
    
    # Step 5: Reinforcement Learning
    logger.info("Step 5: Reinforcement Learning Module")
    try:
        rl_module = ReinforcementLearningModule()
        
        # Create environment
        env = rl_module.create_environment(processed_data, vulnerability_model, combined_scenarios)
        logger.info("Created RL environment")
        
        # Train DQN agent
        dqn_agent = rl_module.train_dqn_agent(env)
        logger.info("Trained DQN agent")
        
        # Train PPO agent
        ppo_agent = rl_module.train_ppo_agent(env)
        logger.info("Trained PPO agent")
        
        # Evaluate agents
        agent_performance = rl_module.evaluate_agents([dqn_agent, ppo_agent], env)
        logger.info(f"Evaluated {len(agent_performance) if isinstance(agent_performance, dict) else 'N/A'} RL agents")
        
        # Save trained agents
        rl_module.save_agents([dqn_agent, ppo_agent], os.path.join(output_dir, f"trained_agents_{timestamp}"))
        logger.info("Reinforcement learning step completed successfully")
    except Exception as e:
        logger.error(f"Error in Reinforcement Learning Module: {e}")
        agent_performance = generate_mock_agent_performance()
    
    # Step 6: Visualization and Reporting
    logger.info("Step 6: Visualization and Reporting Module")
    try:
        visualization_module = VisualizationReportingModule()
        
        # Create vulnerability visualizations
        vulnerability_map = visualization_module.create_vulnerability_map(
            map_type="network",
            include_weather=True,
            show_predictions=True,
            output_format="png",
            output_path=os.path.join(output_dir, f"vulnerability_map_{timestamp}.png")
        )
        logger.info(f"Created vulnerability map: {vulnerability_map['file_path'] if isinstance(vulnerability_map, dict) and 'file_path' in vulnerability_map else 'N/A'}")
        
        # Create performance visualizations
        performance_viz = visualization_module.create_performance_visualizations(
            include_models=["failure_prediction", "rl_agents"],
            metrics=["accuracy", "reward", "outage_reduction"],
            comparison_type="bar_chart",
            output_format="png",
            output_dir=output_dir
        )
        logger.info("Created performance visualizations")
        
        # Generate reports
        report = visualization_module.generate_report(
            report_type="daily_summary",
            include_sections=["summary", "vulnerability", "predictions", "policies"],
            output_format="html",
            output_path=os.path.join(output_dir, f"summary_report_{timestamp}.html")
        )
        logger.info(f"Generated summary report: {report['file_path'] if isinstance(report, dict) and 'file_path' in report else 'N/A'}")
        
        # Launch dashboard (commented out as it would block the script)
        # dashboard = visualization_module.launch_dashboard(dashboard_type="operational")
        # logger.info(f"Launched dashboard at: {dashboard['url'] if isinstance(dashboard, dict) and 'url' in dashboard else 'N/A'}")
        
        logger.info("Visualization and reporting step completed successfully")
    except Exception as e:
        logger.error(f"Error in Visualization and Reporting Module: {e}")
    
    logger.info("GFMF Integration Demo completed successfully")
    logger.info(f"All results saved to: {output_dir}")
    print(f"\nGFMF Integration Demo completed. Results saved to: {output_dir}\n")


def generate_mock_data():
    """Generate mock data for demonstration purposes."""
    logger.info("Generating mock grid and environmental data")
    
    # Create simple grid structure
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
    
    # Create weather dataframe
    dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
    weather_data = pd.DataFrame({
        'date': dates,
        'temperature': 15 + 10 * np.random.rand(90),
        'precipitation': 5 * np.random.rand(90),
        'humidity': 40 + 40 * np.random.rand(90),
        'wind_speed': 15 * np.random.rand(90)
    })
    
    # Create outage dataframe
    outage_data = pd.DataFrame({
        'component_id': [f"transformer_{np.random.randint(0, 10)}" for _ in range(20)] +
                         [f"line_{np.random.randint(0, 9)*2}" for _ in range(15)],
        'start_time': [dates[np.random.randint(0, 85)] for _ in range(35)],
        'end_time': [dates[np.random.randint(5, 90)] for _ in range(35)],
        'cause': np.random.choice(['weather', 'equipment_failure', 'human_error'], 35)
    })
    
    # Combine into processed data
    processed_data = {
        'grid_data': grid_data,
        'weather_data': weather_data,
        'outage_data': outage_data,
        'features': pd.DataFrame({
            'component_id': [comp['id'] for comp in grid_data['components']],
            'age': np.random.randint(1, 30, len(grid_data['components'])),
            'maintenance_frequency': np.random.randint(1, 12, len(grid_data['components']))
        })
    }
    
    return processed_data


def generate_mock_vulnerability_model():
    """Generate mock vulnerability model."""
    logger.info("Generating mock vulnerability model")
    
    # Create simple vulnerability model
    vulnerability_model = {
        "component_scores": {
            f"transformer_{i}": {
                "score": round(0.2 + 0.6 * np.random.rand(), 2),
                "factors": {
                    "age": round(0.1 + 0.4 * np.random.rand(), 2),
                    "weather_exposure": round(0.1 + 0.4 * np.random.rand(), 2),
                    "load": round(0.1 + 0.4 * np.random.rand(), 2)
                }
            } for i in range(10)
        },
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
    
    # Add transmission line vulnerabilities
    for i in range(0, 18, 2):
        vulnerability_model["component_scores"][f"line_{i}"] = {
            "score": round(0.2 + 0.6 * np.random.rand(), 2),
            "factors": {
                "length": round(0.1 + 0.4 * np.random.rand(), 2),
                "weather_exposure": round(0.1 + 0.4 * np.random.rand(), 2),
                "load": round(0.1 + 0.4 * np.random.rand(), 2)
            }
        }
    
    return vulnerability_model


def generate_mock_predictions():
    """Generate mock failure predictions."""
    logger.info("Generating mock failure predictions")
    
    # Create simple prediction results
    predictions = {
        "component_predictions": {
            f"transformer_{i}": {
                "failure_probability": round(0.05 + 0.3 * np.random.rand(), 3),
                "confidence": round(0.5 + 0.5 * np.random.rand(), 2),
                "time_horizon": "7_days"
            } for i in range(10)
        },
        "line_predictions": {
            f"line_{i}": {
                "failure_probability": round(0.05 + 0.2 * np.random.rand(), 3),
                "confidence": round(0.5 + 0.5 * np.random.rand(), 2),
                "time_horizon": "7_days"
            } for i in range(0, 18, 2)
        },
        "metadata": {
            "prediction_timestamp": datetime.now().isoformat(),
            "model_version": "1.0"
        }
    }
    
    # Identify high-risk components
    high_risk = []
    
    for comp_id, pred in predictions["component_predictions"].items():
        if pred["failure_probability"] > 0.25:
            high_risk.append(comp_id)
    
    for line_id, pred in predictions["line_predictions"].items():
        if pred["failure_probability"] > 0.2:
            high_risk.append(line_id)
    
    predictions["high_risk_components"] = high_risk
    
    return predictions


def generate_mock_scenarios():
    """Generate mock scenarios for demonstration."""
    logger.info("Generating mock scenarios")
    
    # Create simple scenarios
    normal_scenarios = [
        {
            "id": f"normal_{i}",
            "weather_conditions": {
                "temperature": 20 + 5 * np.random.rand(),
                "precipitation": 2 * np.random.rand(),
                "humidity": 50 + 10 * np.random.rand(),
                "wind_speed": 5 + 5 * np.random.rand()
            },
            "load_profile": {
                "morning_peak": 60 + 10 * np.random.rand(),
                "evening_peak": 80 + 15 * np.random.rand(),
                "base_load": 40 + 10 * np.random.rand()
            },
            "component_failures": []
        } for i in range(5)
    ]
    
    extreme_scenarios = [
        {
            "id": f"extreme_{i}",
            "weather_conditions": {
                "temperature": 35 + 10 * np.random.rand() if i % 2 == 0 else -10 - 15 * np.random.rand(),
                "precipitation": 20 * np.random.rand() if i % 3 == 0 else 1 * np.random.rand(),
                "humidity": 80 + 15 * np.random.rand() if i % 2 == 1 else 20 + 10 * np.random.rand(),
                "wind_speed": 25 + 15 * np.random.rand() if i % 3 == 2 else 3 + 3 * np.random.rand()
            },
            "load_profile": {
                "morning_peak": 90 + 20 * np.random.rand(),
                "evening_peak": 120 + 20 * np.random.rand(),
                "base_load": 60 + 15 * np.random.rand()
            },
            "component_failures": [
                {"id": f"transformer_{np.random.randint(0, 10)}", "time": np.random.randint(0, 24)},
                {"id": f"line_{np.random.randint(0, 9)*2}", "time": np.random.randint(0, 24)}
            ]
        } for i in range(3)
    ]
    
    combined_scenarios = [
        {
            "id": f"compound_{i}",
            "weather_conditions": {
                "temperature": 30 + 8 * np.random.rand(),
                "precipitation": 15 * np.random.rand(),
                "humidity": 70 + 20 * np.random.rand(),
                "wind_speed": 20 + 10 * np.random.rand()
            },
            "load_profile": {
                "morning_peak": 100 + 20 * np.random.rand(),
                "evening_peak": 130 + 30 * np.random.rand(),
                "base_load": 70 + 15 * np.random.rand()
            },
            "component_failures": [
                {"id": f"transformer_{np.random.randint(0, 10)}", "time": np.random.randint(0, 12)},
                {"id": f"transformer_{np.random.randint(0, 10)}", "time": np.random.randint(12, 24)},
                {"id": f"line_{np.random.randint(0, 9)*2}", "time": np.random.randint(0, 8)},
                {"id": f"line_{np.random.randint(0, 9)*2}", "time": np.random.randint(8, 16)},
                {"id": f"line_{np.random.randint(0, 9)*2}", "time": np.random.randint(16, 24)}
            ],
            "cascade_potential": "high"
        } for i in range(2)
    ]
    
    return normal_scenarios, extreme_scenarios, combined_scenarios


def generate_mock_agent_performance():
    """Generate mock reinforcement learning agent performance metrics."""
    logger.info("Generating mock RL agent performance metrics")
    
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
