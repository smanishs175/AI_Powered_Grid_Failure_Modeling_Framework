"""
Performance visualization component for the Grid Failure Modeling Framework.

This module provides classes and functions for visualizing model performance
metrics, agent learning curves, and comparative analyses.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix


class PerformanceVisualization:
    """
    Class for creating performance visualizations for models and agents.
    
    This class provides methods for visualizing various performance metrics,
    including ROC curves, confusion matrices, learning curves, and comparative
    analyses.
    """
    
    def __init__(self, config=None):
        """
        Initialize the PerformanceVisualization class.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Performance Visualization component")
        
        # Set default configuration if not provided
        self.config = config or {}
        
        # Set visualization parameters
        self.fig_size = self.config.get('figure_size', [10, 6])
        self.dpi = self.config.get('dpi', 100)
        self.style = self.config.get('style', 'whitegrid')
        self.palette = self.config.get('palette', 'deep')
        
        # Initialize figure style
        sns.set_style(self.style)
        sns.set_palette(self.palette)
    
    def create_visualizations(self, models=None, metrics=None, comparison_type='line_chart',
                              output_format='png', output_dir=None, filename_prefix=None):
        """
        Create performance visualizations for the specified models and metrics.
        
        Args:
            models (list, optional): List of models to include in visualizations.
            metrics (list, optional): List of metrics to visualize.
            comparison_type (str): Type of comparison visualization.
            output_format (str): Output format for visualizations.
            output_dir (str, optional): Directory to save visualizations.
            filename_prefix (str, optional): Prefix for output filenames.
            
        Returns:
            dict: Dictionary with visualization metadata and paths.
        """
        # Set default models and metrics if not provided
        if models is None:
            models = ['failure_prediction', 'rl_agents']
        
        if metrics is None:
            metrics = ['accuracy', 'reward', 'outage_reduction']
        
        # Generate default output directory if not provided
        if output_dir is None:
            output_dir = 'outputs/visualization_reporting/performance_visualizations'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate default filename prefix if not provided
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"performance_{timestamp}"
        
        results = {
            'prediction_performance': {},
            'agent_performance': {}
        }
        
        # Create prediction model performance visualizations
        if 'failure_prediction' in models:
            if 'accuracy' in metrics:
                confusion_result = self._create_confusion_matrix_visualization(
                    output_dir=output_dir,
                    filename=f"{filename_prefix}_confusion_matrix.{output_format}",
                    output_format=output_format
                )
                results['prediction_performance']['confusion_matrix'] = confusion_result
            
            if any(m in metrics for m in ['auc', 'roc']):
                roc_result = self._create_roc_curve_visualization(
                    output_dir=output_dir,
                    filename=f"{filename_prefix}_roc_curve.{output_format}",
                    output_format=output_format
                )
                results['prediction_performance']['roc_curve'] = roc_result
        
        # Create RL agent performance visualizations
        if 'rl_agents' in models:
            if any(m in metrics for m in ['reward', 'learning']):
                learning_curves_result = self._create_learning_curves_visualization(
                    output_dir=output_dir,
                    filename=f"{filename_prefix}_learning_curves.{output_format}",
                    output_format=output_format
                )
                results['agent_performance']['learning_curves'] = learning_curves_result
            
            if 'outage_reduction' in metrics:
                outage_result = self._create_comparative_bar_chart(
                    metric='outage_reduction',
                    output_dir=output_dir,
                    filename=f"{filename_prefix}_outage_reduction.{output_format}",
                    output_format=output_format
                )
                results['agent_performance']['outage_reduction'] = outage_result
        
        # If requested, create a combined comparison visualization
        if comparison_type == 'combined':
            combined_result = self._create_combined_comparison(
                models=models,
                metrics=metrics,
                output_dir=output_dir,
                filename=f"{filename_prefix}_combined_comparison.{output_format}",
                output_format=output_format
            )
            results['combined_comparison'] = combined_result
        
        return results
    
    def get_performance_data(self):
        """
        Get performance data for visualization.
        
        Returns:
            dict: Dictionary with performance data.
        """
        return self._load_performance_data()
    
    def get_agent_learning_data(self):
        """
        Get agent learning data for visualization.
        
        Returns:
            dict: Dictionary with agent learning data.
        """
        return self._load_agent_learning_data()
    
    def plot_performance_comparison(self, performance_data, ax=None):
        """
        Plot performance comparison on a given axis.
        
        Args:
            performance_data (dict): Performance data.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
            
        Returns:
            matplotlib.axes.Axes: The axis with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Extract metrics
        metrics = performance_data.get('metrics', {})
        agent_names = list(metrics.keys())
        metric_names = list(metrics[agent_names[0]].keys()) if agent_names else []
        
        # Create bar positions
        x = np.arange(len(metric_names))
        width = 0.8 / len(agent_names)
        
        # Plot bars for each agent
        for i, agent in enumerate(agent_names):
            agent_metrics = [metrics[agent].get(metric, 0) for metric in metric_names]
            ax.bar(x + (i - len(agent_names)/2 + 0.5) * width, agent_metrics, 
                   width, label=agent)
        
        # Set axis labels and legend
        ax.set_ylabel('Performance')
        ax.set_title('Agent Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        
        return ax
    
    def plot_learning_curves(self, learning_data, ax=None):
        """
        Plot learning curves on a given axis.
        
        Args:
            learning_data (dict): Learning curve data.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
            
        Returns:
            matplotlib.axes.Axes: The axis with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size)
        
        agents = learning_data.get('agents', [])
        data = learning_data.get('data', {})
        
        for agent in agents:
            if agent in data:
                episodes = [point[0] for point in data[agent]]
                rewards = [point[1] for point in data[agent]]
                ax.plot(episodes, rewards, label=agent)
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Reward')
        ax.set_title('Agent Learning Curves')
        ax.legend()
        ax.grid(True)
        
        return ax
    
    def _load_performance_data(self):
        """
        Load performance data for visualization.
        
        Returns:
            dict: Dictionary with performance metrics.
        """
        # In a real implementation, this would load data from previous modules
        # For now, we'll create mock data
        try:
            # Try to load from Module 5 output
            metrics_filepath = 'data/reinforcement_learning/performance_metrics/agent_metrics.npy'
            if os.path.exists(metrics_filepath):
                metrics_data = np.load(metrics_filepath, allow_pickle=True).item()
                self.logger.info("Loaded performance metrics from Module 5 output")
                return metrics_data
        except Exception as e:
            self.logger.warning(f"Could not load performance metrics from Module 5: {e}")
        
        # If loading fails, generate mock data
        self.logger.info("Generating mock performance data")
        
        # Create mock metrics for different agents
        agents = ['DQN', 'PPO', 'SAC', 'TD3', 'GAIL']
        metrics = {}
        
        for agent in agents:
            metrics[agent] = {
                'reward': np.random.uniform(0.6, 0.9),
                'outage_reduction': np.random.uniform(0.5, 0.8),
                'stability': np.random.uniform(0.7, 0.95)
            }
        
        return {
            'metrics': metrics,
            'best_agent': max(agents, key=lambda a: metrics[a]['reward']),
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_agent_learning_data(self):
        """
        Load agent learning data for visualization.
        
        Returns:
            dict: Dictionary with agent learning curves.
        """
        # In a real implementation, this would load data from Module 5
        # For now, we'll create mock data
        try:
            # Try to load from Module 5 output
            learning_filepath = 'data/reinforcement_learning/agent_comparisons/learning_curves.npy'
            if os.path.exists(learning_filepath):
                learning_data = np.load(learning_filepath, allow_pickle=True).item()
                self.logger.info("Loaded learning curves from Module 5 output")
                return learning_data
        except Exception as e:
            self.logger.warning(f"Could not load learning curves from Module 5: {e}")
        
        # If loading fails, generate mock data
        self.logger.info("Generating mock learning curve data")
        
        # Generate mock learning curves for different agents
        agents = ['DQN', 'PPO', 'SAC', 'TD3', 'GAIL']
        data = {}
        
        for agent in agents:
            # Generate learning curve with improvement over time
            episodes = np.arange(0, 1000, 10)
            base_reward = np.random.uniform(-10, -5)
            final_reward = np.random.uniform(5, 15)
            
            # Create curve with randomness
            curve = np.linspace(base_reward, final_reward, len(episodes))
            noise = np.random.normal(0, 1, len(episodes))
            rewards = curve + noise
            
            # Smooth with rolling average
            smooth_rewards = np.convolve(rewards, np.ones(5)/5, mode='same')
            
            data[agent] = list(zip(episodes.tolist(), smooth_rewards.tolist()))
        
        return {
            'agents': agents,
            'data': data,
            'best_agent': max(agents, key=lambda a: data[a][-1][1]),
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_prediction_performance_data(self):
        """
        Load prediction model performance data for visualization.
        
        Returns:
            dict: Dictionary with prediction performance metrics.
        """
        # In a real implementation, this would load data from Module 3
        # For now, we'll create mock data
        try:
            # Try to load from Module 3 output
            prediction_filepath = 'data/failure_prediction/model_performance.npy'
            if os.path.exists(prediction_filepath):
                prediction_data = np.load(prediction_filepath, allow_pickle=True).item()
                self.logger.info("Loaded prediction performance from Module 3 output")
                return prediction_data
        except Exception as e:
            self.logger.warning(f"Could not load prediction performance from Module 3: {e}")
        
        # If loading fails, generate mock data
        self.logger.info("Generating mock prediction performance data")
        
        # Generate mock ROC curve data
        n_points = 100
        fpr = np.sort(np.random.uniform(0, 1, n_points))
        tpr = np.sort(np.random.uniform(0, 1, n_points))
        thresholds = np.sort(np.random.uniform(0, 1, n_points))[::-1]
        
        # Ensure the ROC curve is above the diagonal (better than random)
        for i in range(n_points):
            tpr[i] = max(tpr[i], fpr[i] + np.random.uniform(0, 0.2))
            tpr[i] = min(tpr[i], 1.0)
        
        # Calculate AUC
        auc_score = np.trapz(tpr, fpr)
        
        # Generate mock confusion matrix
        tn = np.random.randint(80, 120)
        fp = np.random.randint(10, 30)
        fn = np.random.randint(10, 30)
        tp = np.random.randint(80, 120)
        conf_matrix = np.array([[tn, fp], [fn, tp]])
        
        # Calculate metrics
        total = tn + fp + fn + tp
        accuracy = (tn + tp) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': auc_score
            },
            'confusion_matrix': {
                'matrix': conf_matrix.tolist(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_roc_curve_visualization(self, output_dir=None, filename=None, output_format='png'):
        """
        Create ROC curve visualization for prediction models.
        
        Args:
            output_dir (str, optional): Directory to save the visualization.
            filename (str, optional): Filename for the output file.
            output_format (str): Output format.
            
        Returns:
            dict: Dictionary with visualization metadata.
        """
        # Load prediction performance data
        prediction_data = self._load_prediction_performance_data()
        roc_data = prediction_data.get('roc_curve', {})
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot ROC curve
        fpr = roc_data.get('fpr', [0, 1])
        tpr = roc_data.get('tpr', [0, 1])
        auc_score = roc_data.get('auc', 0.5)
        
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
        
        # Set axis labels and title
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        
        # Set axis limits and grid
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True)
        ax.legend(loc='lower right')
        
        # Save figure if output_dir and filename are provided
        output_path = None
        if output_dir and filename:
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Prepare result
        result = {
            'filename': filename,
            'file_path': output_path,
            'auc_score': auc_score,
            'data_points': list(zip(fpr, tpr)),
            'thresholds': roc_data.get('thresholds', []),
            'format': output_format,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _create_confusion_matrix_visualization(self, output_dir=None, filename=None, output_format='png'):
        """
        Create confusion matrix visualization for prediction models.
        
        Args:
            output_dir (str, optional): Directory to save the visualization.
            filename (str, optional): Filename for the output file.
            output_format (str): Output format.
            
        Returns:
            dict: Dictionary with visualization metadata.
        """
        # Load prediction performance data
        prediction_data = self._load_prediction_performance_data()
        cm_data = prediction_data.get('confusion_matrix', {})
        
        # Extract confusion matrix and metrics
        cm = np.array(cm_data.get('matrix', [[0, 0], [0, 0]]))
        accuracy = cm_data.get('accuracy', 0)
        precision = cm_data.get('precision', 0)
        recall = cm_data.get('recall', 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        # Set axis labels and title
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        # Set tick labels
        ax.set_xticklabels(['No Failure', 'Failure'])
        ax.set_yticklabels(['No Failure', 'Failure'])
        
        # Add metrics text
        plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}',
                   ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save figure if output_dir and filename are provided
        output_path = None
        if output_dir and filename:
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Prepare result
        result = {
            'filename': filename,
            'file_path': output_path,
            'matrix': cm.tolist(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'format': output_format,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _create_learning_curves_visualization(self, output_dir=None, filename=None, output_format='png'):
        """
        Create learning curves visualization for RL agents.
        
        Args:
            output_dir (str, optional): Directory to save the visualization.
            filename (str, optional): Filename for the output file.
            output_format (str): Output format.
            
        Returns:
            dict: Dictionary with visualization metadata.
        """
        # Load agent learning data
        learning_data = self._load_agent_learning_data()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot learning curves for each agent
        agents = learning_data.get('agents', [])
        data = learning_data.get('data', {})
        best_agent = learning_data.get('best_agent', None)
        
        for agent in agents:
            if agent in data:
                episodes = [point[0] for point in data[agent]]
                rewards = [point[1] for point in data[agent]]
                
                linestyle = '-' if agent == best_agent else '--'
                linewidth = 2 if agent == best_agent else 1.5
                
                ax.plot(episodes, rewards, label=agent, linestyle=linestyle, linewidth=linewidth)
        
        # Set axis labels and title
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Reward')
        ax.set_title('Agent Learning Curves')
        
        # Add grid and legend
        ax.grid(True)
        ax.legend(loc='lower right')
        
        # Save figure if output_dir and filename are provided
        output_path = None
        if output_dir and filename:
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Prepare result
        result = {
            'filename': filename,
            'file_path': output_path,
            'agents': agents,
            'data': data,
            'best_agent': best_agent,
            'format': output_format,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _create_comparative_bar_chart(self, metric='outage_reduction', output_dir=None, 
                                     filename=None, output_format='png'):
        """
        Create comparative bar chart for a specific metric across agents.
        
        Args:
            metric (str): Metric to visualize.
            output_dir (str, optional): Directory to save the visualization.
            filename (str, optional): Filename for the output file.
            output_format (str): Output format.
            
        Returns:
            dict: Dictionary with visualization metadata.
        """
        # Load performance data
        performance_data = self._load_performance_data()
        metrics_data = performance_data.get('metrics', {})
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Extract agents and metric values
        agents = list(metrics_data.keys())
        values = [metrics_data[agent].get(metric, 0) for agent in agents]
        
        # Sort by metric value
        sorted_indices = np.argsort(values)[::-1]  # Descending order
        sorted_agents = [agents[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Create color palette
        colors = sns.color_palette(self.palette, len(agents))
        
        # Plot bar chart
        bars = ax.bar(sorted_agents, sorted_values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Set axis labels and title
        metric_name = metric.replace('_', ' ').title()
        ax.set_xlabel('Agent')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Agent Comparison: {metric_name}')
        
        # Set y-axis limits
        ax.set_ylim([0, max(sorted_values) * 1.15])
        
        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure if output_dir and filename are provided
        output_path = None
        if output_dir and filename:
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Prepare result
        result = {
            'filename': filename,
            'file_path': output_path,
            'metric': metric,
            'agents': sorted_agents,
            'values': sorted_values,
            'best_agent': sorted_agents[0] if sorted_agents else None,
            'format': output_format,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _create_combined_comparison(self, models, metrics, output_dir=None, 
                                   filename=None, output_format='png'):
        """
        Create combined comparison visualization with multiple metrics.
        
        Args:
            models (list): List of models to include.
            metrics (list): List of metrics to visualize.
            output_dir (str, optional): Directory to save the visualization.
            filename (str, optional): Filename for the output file.
            output_format (str): Output format.
            
        Returns:
            dict: Dictionary with visualization metadata.
        """
        # Load performance data
        performance_data = self._load_performance_data()
        metrics_data = performance_data.get('metrics', {})
        
        # Create figure with multiple subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(self.fig_size[0] * n_metrics, self.fig_size[1]))
        
        # Ensure axes is a list even if there's only one subplot
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric in a separate subplot
        for i, metric in enumerate(metrics):
            # Extract agents and metric values
            agents = list(metrics_data.keys())
            values = [metrics_data[agent].get(metric, 0) for agent in agents]
            
            # Sort by metric value
            sorted_indices = np.argsort(values)[::-1]  # Descending order
            sorted_agents = [agents[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            # Create color palette
            colors = sns.color_palette(self.palette, len(agents))
            
            # Plot bar chart
            bars = axes[i].bar(sorted_agents, sorted_values, color=colors)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Set axis labels and title
            metric_name = metric.replace('_', ' ').title()
            axes[i].set_xlabel('Agent')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'{metric_name}')
            
            # Set y-axis limits
            axes[i].set_ylim([0, max(sorted_values) * 1.15])
            
            # Add grid lines
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if output_dir and filename are provided
        output_path = None
        if output_dir and filename:
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Prepare result
        result = {
            'filename': filename,
            'file_path': output_path,
            'metrics': metrics,
            'models': models,
            'agents': list(metrics_data.keys()),
            'format': output_format,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
