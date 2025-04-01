"""
Policy optimization module for the Reinforcement Learning component.

This module provides utilities for optimizing and evaluating policies
across different types of scenarios.
"""

import os
import logging
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import yaml

class PolicyOptimizer:
    """
    Handles policy optimization across different scenarios and agents.
    
    This class provides functionality for training, evaluating, and comparing
    different RL algorithms across various grid scenarios.
    """
    
    def __init__(self, config=None):
        """
        Initialize the policy optimizer.
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Policy Optimizer")
        
        # Default configuration
        default_config = {
            'training_steps': 100000,
            'test_scenarios': 50,
            'evaluation_metrics': ['reward', 'outage_rate', 'load_shedding', 'stability', 'recovery_time'],
            'output_dir': 'outputs/reinforcement_learning',
            'save_frequency': 10000,
            'eval_frequency': 5000,
            'verbose': True
        }
        
        # Update with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialize metrics storage
        self.metrics = defaultdict(dict)
        self.training_history = defaultdict(list)
        
        # Make sure output directory exists
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def train_agent(self, agent, env, scenario_type, agent_name, steps=None, save_path=None):
        """
        Train an agent on a specific scenario type.
        
        Args:
            agent: RL agent to train
            env: Environment to train on
            scenario_type (str): Type of scenario (normal, extreme, compound)
            agent_name (str): Name of the agent/algorithm
            steps (int, optional): Number of training steps
            save_path (str, optional): Path to save trained model
            
        Returns:
            dict: Training metrics
        """
        if steps is None:
            steps = self.config['training_steps']
            
        self.logger.info(f"Training {agent_name} on {scenario_type} scenarios for {steps} steps")
        
        # Calculate episodes based on steps
        # Assume average episode length is 100 steps
        avg_episode_len = 100
        num_episodes = max(1, steps // avg_episode_len)
        
        # Train the agent
        training_metrics = agent.train(
            env, 
            num_episodes=num_episodes, 
            max_steps=avg_episode_len,
            eval_freq=max(1, num_episodes // 10)  # Evaluate 10 times during training
        )
        
        # Store metrics
        self.training_history[f"{agent_name}_{scenario_type}"] = training_metrics
        
        # Save model if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save_policy(save_path)
            self.logger.info(f"Saved trained {agent_name} model to {save_path}")
        
        return training_metrics
    
    def evaluate_agent(self, agent, env, scenario_type, agent_name, num_episodes=None):
        """
        Evaluate an agent on a specific scenario type.
        
        Args:
            agent: RL agent to evaluate
            env: Environment to evaluate on
            scenario_type (str): Type of scenario (normal, extreme, compound)
            agent_name (str): Name of the agent/algorithm
            num_episodes (int, optional): Number of evaluation episodes
            
        Returns:
            dict: Evaluation metrics
        """
        if num_episodes is None:
            num_episodes = self.config['test_scenarios']
            
        self.logger.info(f"Evaluating {agent_name} on {scenario_type} scenarios for {num_episodes} episodes")
        
        # Initialize metrics
        metrics = {
            'reward': [],
            'outage_rate': [],
            'load_shedding': [],
            'stability': [],
            'recovery_time': []
        }
        
        # Run evaluation episodes
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Additional metrics
            outages = 0
            load_shedding = 0
            stability_duration = 0
            recovery_steps = 0
            
            while not done:
                action = agent.act(state, eval_mode=True)
                
                # Handle different return formats from different agents
                if isinstance(action, tuple):
                    action = action[0]
                
                next_state, reward, done, info = env.step(action)
                
                # Update metrics from info if available
                if 'outage' in info:
                    outages += info['outage']
                if 'load_shedding' in info:
                    load_shedding += info['load_shedding']
                if 'stability' in info:
                    stability_duration += info['stability']
                if 'recovery_step' in info:
                    recovery_steps = max(recovery_steps, info['recovery_step'])
                
                state = next_state
                episode_reward += reward
            
            # Store episode metrics
            metrics['reward'].append(episode_reward)
            metrics['outage_rate'].append(outages)
            metrics['load_shedding'].append(load_shedding)
            metrics['stability'].append(stability_duration)
            metrics['recovery_time'].append(recovery_steps)
        
        # Calculate summary statistics
        summary = {
            'avg_reward': np.mean(metrics['reward']),
            'std_reward': np.std(metrics['reward']),
            'avg_outage_rate': np.mean(metrics['outage_rate']),
            'avg_load_shedding': np.mean(metrics['load_shedding']),
            'avg_stability': np.mean(metrics['stability']),
            'avg_recovery_time': np.mean(metrics['recovery_time'])
        }
        
        # Store in metrics dictionary
        self.metrics[f"{agent_name}_{scenario_type}"] = summary
        
        return summary
    
    def compare_agents(self, scenario_types=None, agent_names=None, metrics=None):
        """
        Compare multiple agents across different scenario types.
        
        Args:
            scenario_types (list, optional): List of scenario types to compare
            agent_names (list, optional): List of agent names to compare
            metrics (list, optional): List of metrics to compare
            
        Returns:
            pandas.DataFrame: Comparison table
        """
        if scenario_types is None:
            scenario_types = list(set([key.split('_')[-1] for key in self.metrics.keys()]))
            
        if agent_names is None:
            agent_names = list(set([key.split('_')[0] for key in self.metrics.keys()]))
            
        if metrics is None:
            metrics = ['avg_reward', 'avg_outage_rate', 'avg_load_shedding', 'avg_stability', 'avg_recovery_time']
        
        # Create comparison dataframe
        comparison_data = []
        
        for agent in agent_names:
            for scenario in scenario_types:
                key = f"{agent}_{scenario}"
                if key in self.metrics:
                    row = {
                        'Agent': agent,
                        'Scenario': scenario
                    }
                    
                    for metric in metrics:
                        if metric in self.metrics[key]:
                            row[metric] = self.metrics[key][metric]
                    
                    comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def plot_training_curves(self, agent_names=None, scenario_types=None, show=False, save_path=None):
        """
        Plot training curves for agents.
        
        Args:
            agent_names (list, optional): List of agent names to plot
            scenario_types (list, optional): List of scenario types to plot
            show (bool): Whether to display the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if agent_names is None and scenario_types is None:
            keys = list(self.training_history.keys())
        else:
            keys = []
            if agent_names and scenario_types:
                for agent in agent_names:
                    for scenario in scenario_types:
                        key = f"{agent}_{scenario}"
                        if key in self.training_history:
                            keys.append(key)
            elif agent_names:
                for agent in agent_names:
                    for key in self.training_history.keys():
                        if key.startswith(f"{agent}_"):
                            keys.append(key)
            elif scenario_types:
                for scenario in scenario_types:
                    for key in self.training_history.keys():
                        if key.endswith(f"_{scenario}"):
                            keys.append(key)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        for key in keys:
            if 'rewards' in self.training_history[key]:
                rewards = self.training_history[key]['rewards']
                plt.plot(rewards, label=key)
                
                # Also plot moving average if enough data points
                if len(rewards) > 10:
                    moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
                    plt.plot(range(9, len(rewards)), moving_avg, linestyle='--', alpha=0.7, label=f"{key} (MA)")
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            
        if show:
            plt.show()
            
        return plt.gcf()
    
    def plot_evaluation_comparison(self, metric='avg_reward', show=False, save_path=None):
        """
        Plot evaluation comparison for a specific metric.
        
        Args:
            metric (str): Metric to plot
            show (bool): Whether to display the plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Get comparison data
        comparison_df = self.compare_agents(metrics=[metric])
        
        # Check if we have data
        if comparison_df.empty or metric not in comparison_df.columns:
            self.logger.warning(f"No data for metric '{metric}' to plot")
            return None
        
        # Create figure
        plt.figure(figsize=(10, 7))
        
        # Plot grouped bar chart
        sns.barplot(x='Agent', y=metric, hue='Scenario', data=comparison_df)
        
        plt.title(f'Comparison of {metric} across Agents and Scenarios')
        plt.xlabel('Agent')
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        if show:
            plt.show()
            
        return plt.gcf()
    
    def save_results(self, output_dir=None):
        """
        Save all results and metrics.
        
        Args:
            output_dir (str, optional): Directory to save results
            
        Returns:
            str: Path to saved results
        """
        if output_dir is None:
            output_dir = self.config['output_dir']
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        # Save comparison as CSV
        comparison_df = self.compare_agents()
        if not comparison_df.empty:
            csv_path = os.path.join(output_dir, 'agent_comparison.csv')
            comparison_df.to_csv(csv_path, index=False)
        
        # Save plots
        self.plot_training_curves(save_path=os.path.join(output_dir, 'training_curves.png'))
        
        for metric in self.config['evaluation_metrics']:
            metric_name = f'avg_{metric}' if not metric.startswith('avg_') else metric
            self.plot_evaluation_comparison(
                metric=metric_name,
                save_path=os.path.join(output_dir, f'{metric_name}_comparison.png')
            )
        
        self.logger.info(f"Saved all results to {output_dir}")
        
        return output_dir
    
    def load_results(self, input_dir):
        """
        Load saved results and metrics.
        
        Args:
            input_dir (str): Directory with saved results
            
        Returns:
            bool: Success status
        """
        metrics_path = os.path.join(input_dir, 'evaluation_metrics.pkl')
        history_path = os.path.join(input_dir, 'training_history.pkl')
        
        if not os.path.exists(metrics_path) or not os.path.exists(history_path):
            self.logger.warning(f"Missing required files in {input_dir}")
            return False
        
        try:
            # Load metrics
            with open(metrics_path, 'rb') as f:
                self.metrics = pickle.load(f)
            
            # Load training history
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
            
            self.logger.info(f"Loaded results from {input_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return False
