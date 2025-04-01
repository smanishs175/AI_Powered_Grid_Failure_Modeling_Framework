"""
Test script for the Visualization and Reporting Module.

This script demonstrates the basic functionality of the Visualization and Reporting Module,
creating various visualizations, dashboards, and reports.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the module is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gfmf.visualization_reporting.visualization_reporting_module import VisualizationReportingModule
from gfmf.visualization_reporting.grid_visualization import GridVisualization
from gfmf.visualization_reporting.performance_visualization import PerformanceVisualization
from gfmf.visualization_reporting.report_generator import ReportGenerator


def test_grid_visualization():
    """Test the grid visualization component."""
    logger.info("Testing Grid Visualization Component")
    
    # Create output directory
    output_dir = 'outputs/visualization_reporting/tests/grid_viz'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the grid visualization component
    grid_viz = GridVisualization()
    
    # Test different map types
    map_types = ['network', 'heatmap', 'geographic']
    
    for map_type in map_types:
        logger.info(f"Creating {map_type} visualization")
        result = grid_viz.create_vulnerability_map(
            map_type=map_type,
            include_weather=True,
            show_predictions=True,
            output_format='png',
            output_path=f"{output_dir}/vulnerability_map_{map_type}.png"
        )
        
        logger.info(f"Created {map_type} visualization: {result['file_path']}")
    
    return True


def test_performance_visualization():
    """Test the performance visualization component."""
    logger.info("Testing Performance Visualization Component")
    
    # Create output directory
    output_dir = 'outputs/visualization_reporting/tests/performance_viz'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the performance visualization component
    perf_viz = PerformanceVisualization()
    
    # Test different visualization types
    logger.info("Creating ROC curve visualization")
    roc_result = perf_viz._create_roc_curve_visualization(
        output_dir=output_dir,
        filename="test_roc_curve.png",
        output_format='png'
    )
    logger.info(f"Created ROC curve visualization: {roc_result['file_path']}")
    
    logger.info("Creating confusion matrix visualization")
    cm_result = perf_viz._create_confusion_matrix_visualization(
        output_dir=output_dir,
        filename="test_confusion_matrix.png",
        output_format='png'
    )
    logger.info(f"Created confusion matrix visualization: {cm_result['file_path']}")
    
    logger.info("Creating learning curves visualization")
    lc_result = perf_viz._create_learning_curves_visualization(
        output_dir=output_dir,
        filename="test_learning_curves.png",
        output_format='png'
    )
    logger.info(f"Created learning curves visualization: {lc_result['file_path']}")
    
    logger.info("Creating comparative bar chart")
    bar_result = perf_viz._create_comparative_bar_chart(
        metric='outage_reduction',
        output_dir=output_dir,
        filename="test_outage_reduction.png",
        output_format='png'
    )
    logger.info(f"Created comparative bar chart: {bar_result['file_path']}")
    
    return True


def test_report_generator():
    """Test the report generator component."""
    logger.info("Testing Report Generator Component")
    
    # Create output directory
    output_dir = 'outputs/visualization_reporting/tests/reports'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the report generator component
    report_gen = ReportGenerator()
    
    # Test different report types
    report_types = ['daily_summary', 'vulnerability_assessment', 'policy_evaluation']
    
    for report_type in report_types:
        logger.info(f"Generating {report_type} report")
        result = report_gen.generate_report(
            report_type=report_type,
            include_sections=['summary', 'vulnerability', 'predictions', 'policies'],
            output_format='html',
            output_path=f"{output_dir}/{report_type}.html"
        )
        
        logger.info(f"Generated {report_type} report: {result['file_path']}")
    
    return True


def test_visualization_module():
    """Test the full visualization and reporting module."""
    logger.info("Testing Visualization and Reporting Module")
    
    # Initialize the visualization and reporting module
    viz_module = VisualizationReportingModule()
    
    # Test vulnerability map creation
    logger.info("Creating vulnerability map")
    grid_viz = viz_module.create_vulnerability_map(
        map_type='heatmap',
        include_weather=True,
        show_predictions=True,
        output_format='png'
    )
    logger.info(f"Created vulnerability map: {grid_viz['file_path']}")
    
    # Test performance visualizations
    logger.info("Creating performance visualizations")
    performance_viz = viz_module.create_performance_visualizations(
        include_models=['failure_prediction', 'rl_agents'],
        metrics=['accuracy', 'reward', 'outage_reduction'],
        comparison_type='bar_chart',
        output_format='png'
    )
    
    for viz_type, viz_data in performance_viz.items():
        if isinstance(viz_data, dict):
            for name, data in viz_data.items():
                if 'file_path' in data:
                    logger.info(f"Created {viz_type} - {name}: {data['file_path']}")
    
    # Test report generation
    logger.info("Generating report")
    report = viz_module.generate_report(
        report_type='daily_summary',
        include_sections=['summary', 'vulnerability', 'predictions', 'policies'],
        output_format='html'
    )
    logger.info(f"Generated report: {report['file_path']}")
    
    return True


def run_all_tests():
    """Run all tests."""
    tests = [
        test_grid_visualization,
        test_performance_visualization,
        test_report_generator,
        test_visualization_module
    ]
    
    results = []
    
    for test in tests:
        try:
            logger.info(f"Running test: {test.__name__}")
            result = test()
            results.append(result)
            logger.info(f"Test {test.__name__} {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"Test {test.__name__} FAILED with error: {e}")
            results.append(False)
    
    successful = sum(results)
    total = len(results)
    
    logger.info(f"Test Results: {successful}/{total} tests passed")
    
    return successful == total


if __name__ == "__main__":
    logger.info("Starting Visualization and Reporting Module tests")
    success = run_all_tests()
    sys.exit(0 if success else 1)
