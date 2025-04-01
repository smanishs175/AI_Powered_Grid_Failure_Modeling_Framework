"""
Main module interface for the Reinforcement Learning component of the Grid Failure Modeling Framework.

This module integrates environment, agents, and policy optimization components
to provide a complete RL solution for grid failure management.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from gfmf.reinforcement_learning.environment.grid_env import GridEnvironment
from gfmf.reinforcement_learning.agents.dqn_agent import DQNAgent
from gfmf.reinforcement_learning.agents.ppo_agent import PPOAgent
from gfmf.reinforcement_learning.agents.sac_agent import SACAgent
from gfmf.reinforcement_learning.agents.td3_agent import TD3Agent
from gfmf.reinforcement_learning.agents.gail_agent import GAILAgent
from gfmf.reinforcement_learning.policies.policy_optimization import PolicyOptimizer


class ReinforcementLearningModule:
    """
    Main class for the Reinforcement Learning Module.
    
    This class provides a high-level interface for training and evaluating
    RL agents on grid management scenarios, integrating with other modules
    of the GFMF framework.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the RL module with configuration.
        
        Args:
            config_path (str, optional): Path to configuration file. 
                If None, uses default config.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Reinforcement Learning Module")
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            default_config_path = os.path.join(
                os.path.dirname(__file__), 
                'config', 
                'default_config.yaml'
            )
            with open(default_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
        self.logger.info(f"Loaded configuration from {config_path or default_config_path}")
        
        # Set up paths
        self.base_path = self.config.get('base_path', 'data/reinforcement_learning/')
        self.exported_policies_path = os.path.join(self.base_path, 'exported_policies')
        self.training_results_path = os.path.join(self.base_path, 'training_results')
        
        # Create directories if they don't exist
        os.makedirs(self.exported_policies_path, exist_ok=True)
        os.makedirs(self.training_results_path, exist_ok=True)
        
        # Initialize components
        self.environment = None
        self.agents = {}
        self.optimizer = None
        
        self.logger.info("Reinforcement Learning Module initialized")
    
    def load_scenario_data(self, scenario_paths=None):
        """
        Load scenario data from Module 4 outputs.
        
        Args:
            scenario_paths (dict, optional): Dictionary mapping scenario types to file paths.
                If None, uses default paths.
                
        Returns:
            dict: Dictionary of loaded scenarios by type.
        """
        self.logger.info("Loading scenario data from Module 4")
        
        if scenario_paths is None:
            scenario_paths = {
                'normal': 'data/scenario_generation/generated_scenarios/normal_scenarios.pkl',
                'extreme': 'data/scenario_generation/generated_scenarios/extreme_scenarios.pkl',
                'compound': 'data/scenario_generation/generated_scenarios/compound_scenarios.pkl'
            }
        
        scenarios = {}
        
        for scenario_type, path in scenario_paths.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        scenarios[scenario_type] = pickle.load(f)
                    self.logger.info(f"Loaded {len(scenarios[scenario_type])} {scenario_type} scenarios")
                else:
                    self.logger.warning(f"Scenario file not found: {path}")
                    scenarios[scenario_type] = []
            except Exception as e:
                self.logger.error(f"Error loading scenarios from {path}: {e}")
                scenarios[scenario_type] = []
        
        # Load cascade models
        cascade_model_path = 'data/scenario_generation/cascade_models/propagation_models.pkl'
        try:
            if os.path.exists(cascade_model_path):
                with open(cascade_model_path, 'rb') as f:
                    scenarios['cascade_models'] = pickle.load(f)
                self.logger.info(f"Loaded cascade models from {cascade_model_path}")
            else:
                self.logger.warning(f"Cascade model file not found: {cascade_model_path}")
                scenarios['cascade_models'] = {}
        except Exception as e:
            self.logger.error(f"Error loading cascade models from {cascade_model_path}: {e}")
            scenarios['cascade_models'] = {}
            
        return scenarios
    
    def initialize_environment(self, scenarios=None):
        """
        Initialize the RL environment with scenario data.
        
        Args:
            scenarios (dict, optional): Dictionary of scenarios. If None, loads from default paths.
            
        Returns:
            GridEnvironment: The initialized environment.
        """
        self.logger.info("Initializing Grid Environment")
        
        if scenarios is None:
            scenarios = self.load_scenario_data()
            
        env_config = self.config.get('environment', {})
        
        self.environment = GridEnvironment(
            scenarios=scenarios,
            max_steps=env_config.get('max_steps', 100),
            reward_weights=env_config.get('reward_weights', {
                'stability': 1.0,
                'outage': -1.0,
                'action': -0.1
            })
        )
        
        self.logger.info(f"Environment initialized with state_dim={self.environment.observation_space.shape[0]}, action_dim={self.environment.action_space.n}")
        
        return self.environment
    
    def initialize_agents(self, agent_types=None):
        """
        Initialize RL agents according to configuration.
        
        Args:
            agent_types (list, optional): List of agent types to initialize.
                If None, initializes all agents in config.
                
        Returns:
            dict: Dictionary of initialized agents.
        """
        self.logger.info("Initializing RL agents")
        
        if self.environment is None:
            self.initialize_environment()
            
        if agent_types is None:
            agent_types = self.config.get('agents', {}).keys()
            
        state_dim = self.environment.observation_space.shape[0]
        action_dim = self.environment.action_space.n
        
        for agent_type in agent_types:
            agent_config = self.config.get('agents', {}).get(agent_type, {})
            
            try:
                if agent_type == 'dqn':
                    self.agents[agent_type] = DQNAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        config=agent_config
                    )
                elif agent_type == 'ppo':
                    self.agents[agent_type] = PPOAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        config=agent_config
                    )
                elif agent_type == 'sac':
                    self.agents[agent_type] = SACAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        config=agent_config
                    )
                elif agent_type == 'td3':
                    self.agents[agent_type] = TD3Agent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        config=agent_config
                    )
                elif agent_type == 'gail':
                    self.agents[agent_type] = GAILAgent(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        config=agent_config
                    )
                else:
                    self.logger.warning(f"Unknown agent type: {agent_type}")
                    continue
                    
                self.logger.info(f"Initialized {agent_type.upper()} agent")
            except Exception as e:
                self.logger.error(f"Error initializing {agent_type} agent: {e}")
        
        return self.agents
    
    def initialize_optimizer(self):
        """
        Initialize the policy optimizer.
        
        Returns:
            PolicyOptimizer: The initialized optimizer.
        """
        self.logger.info("Initializing Policy Optimizer")
        
        optimizer_config = self.config.get('optimizer', {})
        
        self.optimizer = PolicyOptimizer(
            agents=self.agents,
            environment=self.environment,
            config=optimizer_config
        )
        
        self.logger.info("Policy Optimizer initialized")
        
        return self.optimizer
    
    def train_and_evaluate_agents(self, agents=None, scenario_types=None, 
                                 training_steps=100000, eval_frequency=10000):
        """
        Train and evaluate multiple agents on scenarios.
        
        Args:
            agents (list, optional): List of agent types to train. 
                If None, trains all initialized agents.
            scenario_types (list, optional): List of scenario types to use.
                If None, uses all available types.
            training_steps (int): Number of steps to train each agent.
            eval_frequency (int): How often to evaluate agents during training.
            
        Returns:
            dict: Training results including metrics and policies.
        """
        self.logger.info(f"Starting training and evaluation of agents for {training_steps} steps")
        
        # Initialize components if not already done
        if self.environment is None:
            self.initialize_environment()
        
        if not self.agents:
            if agents is not None:
                self.initialize_agents(agent_types=agents)
            else:
                self.initialize_agents()
                agents = list(self.agents.keys())
        elif agents is None:
            agents = list(self.agents.keys())
            
        if self.optimizer is None:
            self.initialize_optimizer()
        
        # Prepare results dictionary
        results = {
            'training_curves': {},
            'evaluation_metrics': {},
            'best_policies': {},
            'scenario_performance': {}
        }
        
        # Train and evaluate each agent
        for agent_type in agents:
            if agent_type not in self.agents:
                self.logger.warning(f"Agent {agent_type} not initialized, skipping")
                continue
                
            self.logger.info(f"Training {agent_type.upper()} agent")
            
            agent = self.agents[agent_type]
            
            # Train agent
            training_metrics = self.optimizer.train_agent(
                agent=agent,
                agent_type=agent_type,
                steps=training_steps,
                eval_frequency=eval_frequency,
                scenario_types=scenario_types
            )
            
            results['training_curves'][agent_type] = training_metrics['learning_curve']
            
            # Evaluate on test scenarios
            evaluation_metrics = self.optimizer.evaluate_agent(
                agent=agent,
                agent_type=agent_type,
                scenario_types=scenario_types
            )
            
            results['evaluation_metrics'][agent_type] = evaluation_metrics['overall']
            results['scenario_performance'][agent_type] = evaluation_metrics['by_scenario']
            
            # Save best policy
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            policy_path = os.path.join(
                self.exported_policies_path, 
                f"{agent_type}_policy_{timestamp}.pkl"
            )
            
            self.export_policy(agent_type, policy_path)
            
            results['best_policies'][agent_type] = {
                'path': policy_path,
                'metrics': evaluation_metrics['overall']
            }
            
            self.logger.info(f"Completed training and evaluation of {agent_type.upper()} agent")
        
        # Save results
        results_path = os.path.join(
            self.training_results_path,
            f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
            
        self.logger.info(f"Saved training results to {results_path}")
        
        # Determine best overall policy
        best_agent = None
        best_reward = float('-inf')
        
        for agent_type, metrics in results['evaluation_metrics'].items():
            if metrics['average_reward'] > best_reward:
                best_reward = metrics['average_reward']
                best_agent = agent_type
                
        if best_agent:
            results['best_policy'] = {
                'agent': best_agent,
                'average_reward': best_reward,
                'policy': results['best_policies'][best_agent]['path'],
                'outage_reduction': results['evaluation_metrics'][best_agent].get('outage_reduction', 0)
            }
            
            self.logger.info(f"Best performing agent: {best_agent.upper()} with average reward {best_reward:.2f}")
        
        return results
    
    def export_policy(self, agent_type, output_path=None):
        """
        Export a trained policy for deployment.
        
        Args:
            agent_type (str): Type of agent whose policy to export.
            output_path (str, optional): Path to save the policy.
                If None, generates a default path.
                
        Returns:
            str: Path where policy was saved.
        """
        self.logger.info(f"Exporting {agent_type.upper()} policy")
        
        if agent_type not in self.agents:
            raise ValueError(f"Agent {agent_type} not initialized")
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.exported_policies_path,
                f"{agent_type}_policy_{timestamp}.pkl"
            )
            
        self.agents[agent_type].save_policy(output_path)
        self.logger.info(f"Exported policy to {output_path}")
        
        return output_path
    
    def visualize_agent_comparison(self, results, output_dir=None):
        """
        Create visualization comparing agent performance.
        
        Args:
            results (dict): Results from train_and_evaluate_agents.
            output_dir (str, optional): Directory to save visualizations.
                If None, uses default path.
                
        Returns:
            list: Paths to saved visualizations.
        """
        self.logger.info("Creating agent comparison visualizations")
        
        if output_dir is None:
            output_dir = os.path.join('outputs', 'reinforcement_learning')
            
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        # 1. Comparative bar chart of overall performance
        plt.figure(figsize=(12, 8))
        metrics = ['average_reward', 'outage_reduction', 'stability_score', 'response_time']
        available_metrics = set()
        
        for agent_type, agent_metrics in results['evaluation_metrics'].items():
            available_metrics.update(agent_metrics.keys())
            
        metrics = [m for m in metrics if m in available_metrics]
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            agent_types = []
            metric_values = []
            
            for agent_type, agent_metrics in results['evaluation_metrics'].items():
                if metric in agent_metrics:
                    agent_types.append(agent_type.upper())
                    metric_values.append(agent_metrics[metric])
                    
            if metric_values:
                sns.barplot(x=agent_types, y=metric_values)
                plt.title(f'{metric.replace("_", " ").title()}')
                plt.xticks(rotation=45)
                plt.tight_layout()
        
        comparison_path = os.path.join(output_dir, 'agent_comparison.png')
        plt.savefig(comparison_path, dpi=100)
        plt.close()
        saved_paths.append(comparison_path)
        
        # 2. Learning curves
        plt.figure(figsize=(10, 6))
        for agent_type, curve in results['training_curves'].items():
            steps = [point.get('step', i) for i, point in enumerate(curve)]
            rewards = [point.get('reward', 0) for point in curve]
            plt.plot(steps, rewards, label=agent_type.upper())
            
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        curves_path = os.path.join(output_dir, 'learning_curves.png')
        plt.savefig(curves_path, dpi=100)
        plt.close()
        saved_paths.append(curves_path)
        
        # 3. Scenario-specific performance heatmap
        if 'scenario_performance' in results and results['scenario_performance']:
            # Get all scenario types and agents
            scenario_types = set()
            agent_types = list(results['scenario_performance'].keys())
            
            for agent_type, scenario_metrics in results['scenario_performance'].items():
                scenario_types.update(scenario_metrics.keys())
                
            scenario_types = sorted(list(scenario_types))
            
            # Create heatmap data
            heatmap_data = []
            for scenario_type in scenario_types:
                row = []
                for agent_type in agent_types:
                    if (scenario_type in results['scenario_performance'][agent_type] and
                            'average_reward' in results['scenario_performance'][agent_type][scenario_type]):
                        row.append(results['scenario_performance'][agent_type][scenario_type]['average_reward'])
                    else:
                        row.append(float('nan'))
                heatmap_data.append(row)
                
            if heatmap_data:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    data=heatmap_data,
                    annot=True,
                    fmt=".2f",
                    xticklabels=[a.upper() for a in agent_types],
                    yticklabels=scenario_types,
                    cmap='viridis'
                )
                plt.title('Average Reward by Scenario Type and Agent')
                plt.xlabel('Agent')
                plt.ylabel('Scenario Type')
                plt.tight_layout()
                
                heatmap_path = os.path.join(output_dir, 'scenario_performance_heatmap.png')
                plt.savefig(heatmap_path, dpi=100)
                plt.close()
                saved_paths.append(heatmap_path)
        
        self.logger.info(f"Saved {len(saved_paths)} visualizations to {output_dir}")
        
        return saved_paths
    
    def visualize_learning_curves(self, training_results, output_dir=None):
        """
        Visualize learning curves from training.
        
        Args:
            training_results (dict): Results from training.
            output_dir (str, optional): Directory to save visualizations.
                If None, uses default path.
                
        Returns:
            str: Path to saved visualization.
        """
        self.logger.info("Creating learning curve visualization")
        
        if output_dir is None:
            output_dir = os.path.join('outputs', 'reinforcement_learning')
            
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        if 'training_curves' in training_results:
            for agent_type, curve in training_results['training_curves'].items():
                steps = [point.get('step', i) for i, point in enumerate(curve)]
                rewards = [point.get('reward', 0) for point in curve]
                
                if 'loss' in curve[0]:
                    losses = [point.get('loss', 0) for point in curve]
                else:
                    losses = None
                    
                plt.subplot(1, 2 if losses else 1, 1)
                plt.plot(steps, rewards, label=agent_type.upper())
                plt.xlabel('Training Steps')
                plt.ylabel('Average Reward')
                plt.title('Reward During Training')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if losses:
                    plt.subplot(1, 2, 2)
                    plt.plot(steps, losses, label=agent_type.upper())
                    plt.xlabel('Training Steps')
                    plt.ylabel('Loss')
                    plt.title('Loss During Training')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        curves_path = os.path.join(output_dir, 'detailed_learning_curves.png')
        plt.savefig(curves_path, dpi=100)
        plt.close()
        
        self.logger.info(f"Saved learning curve visualization to {curves_path}")
        
        return curves_path
    
    def load_policy(self, policy_path, agent_type=None):
        """
        Load a saved policy.
        
        Args:
            policy_path (str): Path to the saved policy.
            agent_type (str, optional): Type of agent for the policy.
                If None, attempts to determine from file name.
                
        Returns:
            object: Loaded policy.
        """
        self.logger.info(f"Loading policy from {policy_path}")
        
        if agent_type is None:
            # Try to determine agent type from file name
            file_name = os.path.basename(policy_path)
            for known_type in ['dqn', 'ppo', 'sac', 'td3', 'gail']:
                if known_type in file_name.lower():
                    agent_type = known_type
                    break
                    
            if agent_type is None:
                raise ValueError("Could not determine agent type from file name. Please specify.")
        
        if agent_type not in self.agents and self.initialize_environment():
            self.initialize_agents(agent_types=[agent_type])
            
        if agent_type not in self.agents:
            raise ValueError(f"Agent {agent_type} could not be initialized")
            
        policy = self.agents[agent_type].load_policy(policy_path)
        
        self.logger.info(f"Loaded {agent_type.upper()} policy from {policy_path}")
        
        return policy
