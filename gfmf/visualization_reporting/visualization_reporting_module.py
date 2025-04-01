"""
Main module for visualization and reporting in the Grid Failure Modeling Framework.

This module integrates all visualization components and provides a unified interface
for creating visualizations, dashboards, and reports.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from gfmf.visualization_reporting.grid_visualization import GridVisualization
from gfmf.visualization_reporting.performance_visualization import PerformanceVisualization
from gfmf.visualization_reporting.dashboard import Dashboard
from gfmf.visualization_reporting.report_generator import ReportGenerator


class VisualizationReportingModule:
    """
    Main class for the Visualization and Reporting Module.
    
    This class integrates all visualization components and provides methods for
    creating grid visualizations, performance visualizations, dashboards, and reports.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the Visualization and Reporting Module.
        
        Args:
            config_path (str, optional): Path to configuration file.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Visualization and Reporting Module")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize visualization components
        self.grid_viz = GridVisualization(self.config.get('grid_visualization', {}))
        self.performance_viz = PerformanceVisualization(self.config.get('performance_visualization', {}))
        self.dashboard = Dashboard(self.config.get('dashboard', {}))
        self.report_generator = ReportGenerator(self.config.get('report_generator', {}))
        
        # Set up output directories
        self.output_dir = self.config.get('output_dir', 'outputs/visualization_reporting')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sub-directories for different visualization types
        self.grid_viz_dir = os.path.join(self.output_dir, 'grid_visualizations')
        self.performance_viz_dir = os.path.join(self.output_dir, 'performance_visualizations')
        self.dashboard_dir = os.path.join(self.output_dir, 'dashboards')
        self.report_dir = os.path.join(self.output_dir, 'reports')
        
        for directory in [self.grid_viz_dir, self.performance_viz_dir, self.dashboard_dir, self.report_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file.
            
        Returns:
            dict: Configuration dictionary.
        """
        default_config = {
            'output_dir': 'outputs/visualization_reporting',
            'grid_visualization': {
                'node_size': 300,
                'font_size': 10,
                'color_scheme': {
                    'operational': 'green',
                    'at_risk': 'yellow',
                    'failed': 'red'
                },
                'map_style': 'light',
                'default_format': 'png',
                'dpi': 300
            },
            'performance_visualization': {
                'figure_size': [10, 6],
                'dpi': 100,
                'style': 'whitegrid',
                'palette': 'deep',
                'default_format': 'png'
            },
            'dashboard': {
                'port': 8050,
                'theme': 'light',
                'refresh_interval': 300,
                'default_layout': 'grid',
                'max_items_per_page': 6
            },
            'report_generator': {
                'template_dir': 'templates/reports',
                'default_format': 'pdf',
                'logo_path': 'assets/logo.png',
                'company_name': 'Grid Resilience Inc.',
                'default_sections': ['summary', 'vulnerability', 'predictions', 'policies']
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                
                # Deep merge user config with default config
                def merge_dicts(default, user):
                    for key, value in user.items():
                        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                            merge_dicts(default[key], value)
                        else:
                            default[key] = value
                
                merge_dicts(default_config, user_config)
                self.logger.info(f"Loaded configuration from: {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_path}: {e}")
                self.logger.info("Using default configuration")
        else:
            self.logger.info("No configuration file provided or file not found. Using default configuration.")
        
        return default_config
    
    def create_vulnerability_map(self, map_type='heatmap', include_weather=True, 
                                 show_predictions=True, output_format=None):
        """
        Create grid vulnerability visualization.
        
        Args:
            map_type (str): Type of map to create ('heatmap', 'network', 'geographic').
            include_weather (bool): Whether to include weather overlays.
            show_predictions (bool): Whether to show failure predictions.
            output_format (str, optional): Output format ('png', 'svg', 'interactive').
            
        Returns:
            dict: Dictionary with visualization metadata and paths.
        """
        self.logger.info(f"Creating vulnerability map: type={map_type}, weather={include_weather}, predictions={show_predictions}")
        
        # Set output format if not specified
        if output_format is None:
            output_format = self.config['grid_visualization']['default_format']
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vulnerability_map_{map_type}_{timestamp}"
        
        # Create full output path
        output_path = os.path.join(self.grid_viz_dir, filename)
        if output_format != 'interactive':
            output_path = f"{output_path}.{output_format}"
        
        # Create the visualization
        result = self.grid_viz.create_vulnerability_map(
            map_type=map_type,
            include_weather=include_weather,
            show_predictions=show_predictions,
            output_format=output_format,
            output_path=output_path
        )
        
        return result
    
    def create_performance_visualizations(self, include_models=None, metrics=None, 
                                         comparison_type='line_chart', output_format=None):
        """
        Create performance visualizations for models and agents.
        
        Args:
            include_models (list, optional): List of models to include.
            metrics (list, optional): List of metrics to visualize.
            comparison_type (str): Type of comparison visualization.
            output_format (str, optional): Output format.
            
        Returns:
            dict: Dictionary with visualization metadata and paths.
        """
        self.logger.info(f"Creating performance visualizations: models={include_models}, metrics={metrics}")
        
        # Default models and metrics if not specified
        if include_models is None:
            include_models = ['failure_prediction', 'rl_agents']
        
        if metrics is None:
            metrics = ['accuracy', 'reward', 'outage_reduction']
        
        # Set output format if not specified
        if output_format is None:
            output_format = self.config['performance_visualization']['default_format']
        
        # Generate filename prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"performance_{comparison_type}_{timestamp}"
        
        # Create the visualizations
        result = self.performance_viz.create_visualizations(
            models=include_models,
            metrics=metrics,
            comparison_type=comparison_type,
            output_format=output_format,
            output_dir=self.performance_viz_dir,
            filename_prefix=filename_prefix
        )
        
        return result
    
    def launch_dashboard(self, dashboard_type='operational', auto_refresh=True, 
                        refresh_interval=None, components=None):
        """
        Launch an interactive dashboard.
        
        Args:
            dashboard_type (str): Type of dashboard ('operational', 'vulnerability', 'policy').
            auto_refresh (bool): Whether to auto-refresh the dashboard.
            refresh_interval (int, optional): Refresh interval in seconds.
            components (list, optional): List of dashboard components to include.
            
        Returns:
            dict: Dictionary with dashboard metadata and URL.
        """
        self.logger.info(f"Launching dashboard: type={dashboard_type}, auto_refresh={auto_refresh}")
        
        # Set refresh interval if not specified
        if refresh_interval is None:
            refresh_interval = self.config['dashboard']['refresh_interval']
        
        # Launch the dashboard
        result = self.dashboard.launch(
            dashboard_type=dashboard_type,
            auto_refresh=auto_refresh,
            refresh_interval=refresh_interval,
            components=components,
            output_dir=self.dashboard_dir
        )
        
        return result
    
    def generate_report(self, report_type='daily_summary', include_sections=None,
                       output_format=None, output_path=None):
        """
        Generate an automated report.
        
        Args:
            report_type (str): Type of report to generate.
            include_sections (list, optional): List of sections to include in the report.
            output_format (str, optional): Output format for the report.
            output_path (str, optional): Path where the report should be saved.
            
        Returns:
            dict: Dictionary with report metadata and file path.
        """
        self.logger.info(f"Generating report: type={report_type}")
        
        # Set default sections if not specified
        if include_sections is None:
            include_sections = self.config['report_generator']['default_sections']
        
        # Set output format if not specified
        if output_format is None:
            output_format = self.config['report_generator']['default_format']
        
        # Generate default output path if not specified
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_{timestamp}.{output_format}"
            output_path = os.path.join(self.report_dir, filename)
        
        # Generate the report
        result = self.report_generator.generate_report(
            report_type=report_type,
            include_sections=include_sections,
            output_format=output_format,
            output_path=output_path
        )
        
        return result
    
    def create_combined_visualization(self, viz_types, output_path=None, output_format=None):
        """
        Create a combined visualization with multiple visualization types.
        
        Args:
            viz_types (list): List of visualization types to combine.
            output_path (str, optional): Path where the visualization should be saved.
            output_format (str, optional): Output format for the visualization.
            
        Returns:
            dict: Dictionary with visualization metadata and file path.
        """
        self.logger.info(f"Creating combined visualization: types={viz_types}")
        
        # Set output format if not specified
        if output_format is None:
            output_format = self.config['grid_visualization']['default_format']
        
        # Generate default output path if not specified
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"combined_visualization_{timestamp}.{output_format}"
            output_path = os.path.join(self.output_dir, filename)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        result = {
            'file_path': output_path,
            'visualization_types': viz_types,
            'timestamp': datetime.now().isoformat(),
            'components': []
        }
        
        # Add visualizations based on specified types
        for i, viz_type in enumerate(viz_types):
            if viz_type == 'vulnerability_map':
                # Create vulnerability map subplot
                ax = fig.add_subplot(len(viz_types), 1, i+1)
                vulnerability_data = self.grid_viz.get_vulnerability_data()
                self.grid_viz.plot_vulnerability_map(vulnerability_data, ax=ax)
                result['components'].append({
                    'type': 'vulnerability_map',
                    'position': i
                })
            
            elif viz_type == 'performance_comparison':
                # Create performance comparison subplot
                ax = fig.add_subplot(len(viz_types), 1, i+1)
                performance_data = self.performance_viz.get_performance_data()
                self.performance_viz.plot_performance_comparison(performance_data, ax=ax)
                result['components'].append({
                    'type': 'performance_comparison',
                    'position': i
                })
            
            elif viz_type == 'agent_learning_curves':
                # Create agent learning curves subplot
                ax = fig.add_subplot(len(viz_types), 1, i+1)
                learning_data = self.performance_viz.get_agent_learning_data()
                self.performance_viz.plot_learning_curves(learning_data, ax=ax)
                result['components'].append({
                    'type': 'agent_learning_curves',
                    'position': i
                })
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['grid_visualization']['dpi'])
        
        return result
