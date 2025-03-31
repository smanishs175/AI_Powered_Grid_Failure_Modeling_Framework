"""
Unit tests for the Visualization and Reporting Module (Module 6)
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, mock_open

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gfmf.visualization_reporting.visualization_reporting_module import VisualizationReportingModule


class TestVisualizationReportingModule(unittest.TestCase):
    """Test cases for the VisualizationReportingModule class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.vr_module = VisualizationReportingModule()
        
        # Create mock vulnerability scores
        self.vulnerability_scores = pd.DataFrame({
            'component_id': range(1, 11),
            'vulnerability_score': np.random.uniform(0, 1, 10),
            'type': ['line', 'transformer', 'bus', 'line', 'transformer', 
                    'bus', 'line', 'transformer', 'bus', 'line']
        })
        
        # Create mock failure predictions
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        self.failure_predictions = pd.DataFrame({
            'date': np.repeat(dates, 5),
            'component_id': np.tile(range(1, 6), 10),
            'failure_probability': np.random.uniform(0, 1, 50)
        })
        
        # Create mock hardening policy
        self.hardening_policy = {
            'policy_id': 'policy_1',
            'algorithm': 'sac',
            'actions': [
                {'component_id': 1, 'action': 'replace', 'priority': 'high'},
                {'component_id': 3, 'action': 'reinforce', 'priority': 'medium'},
                {'component_id': 5, 'action': 'monitor', 'priority': 'low'}
            ]
        }
        
        # Create mock agent performance metrics
        self.agent_performance = {
            'sac': {
                'mean_reward': 0.8,
                'std_reward': 0.1,
                'success_rate': 0.85,
                'episodes': 10,
                'learning_curve': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            },
            'ppo': {
                'mean_reward': 0.75,
                'std_reward': 0.15,
                'success_rate': 0.8,
                'episodes': 10,
                'learning_curve': [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
            }
        }
        
        # Mock grid topology data
        self.grid_data = pd.DataFrame({
            'component_id': range(1, 11),
            'type': ['line', 'transformer', 'bus', 'line', 'transformer', 
                    'bus', 'line', 'transformer', 'bus', 'line'],
            'from': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'to': [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
            'capacity': np.random.uniform(50, 200, 10),
            'age': np.random.uniform(1, 20, 10),
            'criticality': np.random.uniform(0.1, 1.0, 10)
        })

    def test_initialization(self):
        """Test that module initializes correctly."""
        self.assertIsNotNone(self.vr_module)
        self.assertIsNotNone(self.vr_module.config)
        
    def test_plot_vulnerability_heatmap(self):
        """Test vulnerability heatmap plotting."""
        # Mock plt.figure and plt.savefig to avoid actual plotting
        with patch('matplotlib.pyplot.figure', return_value=MagicMock()):
            with patch('matplotlib.pyplot.savefig'):
                figure = self.vr_module.plot_vulnerability_heatmap(
                    self.vulnerability_scores, save_path='test.png'
                )
                self.assertIsInstance(figure, MagicMock)
                
    def test_plot_failure_prediction_timeline(self):
        """Test failure prediction timeline plotting."""
        # Mock plt.figure and plt.savefig
        with patch('matplotlib.pyplot.figure', return_value=MagicMock()):
            with patch('matplotlib.pyplot.savefig'):
                figure = self.vr_module.plot_failure_prediction_timeline(
                    self.failure_predictions, save_path='test.png'
                )
                self.assertIsInstance(figure, MagicMock)
                
    def test_visualize_grid_topology(self):
        """Test grid topology visualization."""
        # This is more complex and might use networkx, so we'll mock at a higher level
        with patch.object(
            self.vr_module, '_create_network_graph', 
            return_value=MagicMock()
        ):
            with patch('matplotlib.pyplot.figure', return_value=MagicMock()):
                with patch('matplotlib.pyplot.savefig'):
                    figure = self.vr_module.visualize_grid_topology(
                        self.grid_data, self.vulnerability_scores, save_path='test.png'
                    )
                    self.assertIsInstance(figure, MagicMock)
    
    def test_plot_learning_curves(self):
        """Test learning curve plotting."""
        # Mock plt.figure and plt.savefig
        with patch('matplotlib.pyplot.figure', return_value=MagicMock()):
            with patch('matplotlib.pyplot.savefig'):
                figure = self.vr_module.plot_learning_curves(
                    self.agent_performance, save_path='test.png'
                )
                self.assertIsInstance(figure, MagicMock)
    
    def test_generate_policy_summary(self):
        """Test policy summary generation."""
        summary = self.vr_module.generate_policy_summary(
            self.hardening_policy, self.vulnerability_scores
        )
        
        self.assertIsInstance(summary, dict)
        self.assertIn('policy_id', summary)
        self.assertIn('num_actions', summary)
        self.assertIn('priority_breakdown', summary)
        
    def test_generate_html_report(self):
        """Test HTML report generation."""
        # Mock file operations
        with patch('builtins.open', mock_open()):
            # Mock the rendering function
            with patch.object(
                self.vr_module, '_render_html_template', 
                return_value='<html>Mock Report</html>'
            ):
                report_path = self.vr_module.generate_html_report(
                    self.vulnerability_scores,
                    self.failure_predictions,
                    self.hardening_policy,
                    self.agent_performance,
                    output_dir='test_output'
                )
                
                self.assertIsInstance(report_path, str)
                self.assertTrue('html' in report_path.lower())
    
    def test_export_results_to_csv(self):
        """Test CSV export functionality."""
        # Prepare test data
        test_data = {
            'vulnerability_scores': self.vulnerability_scores,
            'failure_predictions': self.failure_predictions
        }
        
        # Mock file operations
        with patch('pandas.DataFrame.to_csv'):
            result = self.vr_module.export_results_to_csv(
                test_data, output_dir='test_output'
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('vulnerability_scores', result)
            self.assertIn('failure_predictions', result)
    
    def test_create_dashboard(self):
        """Test dashboard creation functionality."""
        # This might involve creating a Plotly Dash app or similar
        # We'll mock the creation process
        with patch.object(
            self.vr_module, '_initialize_dashboard', 
            return_value=MagicMock()
        ):
            dashboard = self.vr_module.create_dashboard(
                self.vulnerability_scores,
                self.failure_predictions,
                self.hardening_policy,
                port=8050
            )
            
            self.assertIsNotNone(dashboard)


if __name__ == '__main__':
    unittest.main()
