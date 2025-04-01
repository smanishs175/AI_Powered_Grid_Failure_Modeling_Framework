# Module 6: Visualization and Reporting Module

## Overview

The Visualization and Reporting Module is the final module of the Grid Failure Modeling Framework (GFMF), responsible for creating visual representations of grid vulnerability, model performance metrics, and generating comprehensive reports for decision makers.

This module integrates data and results from all previous modules to provide actionable insights through intuitive visualizations and structured reports.

## Components

### 1. Grid Visualization (`grid_visualization.py`)

Creates visual representations of grid vulnerability and operational status:

- **Network Visualizations**: Displays grid topology with component status
- **Heatmaps**: Shows vulnerability hotspots across the grid
- **Geographic Visualizations**: Maps components to real-world locations with overlays

### 2. Performance Visualization (`performance_visualization.py`)

Visualizes metrics for model performance evaluation:

- **ROC Curves**: For evaluating prediction models
- **Confusion Matrices**: For analyzing classification performance
- **Learning Curves**: For tracking RL agent training progress
- **Comparative Charts**: For comparing algorithm performance

### 3. Dashboard (`dashboard.py`)

Provides interactive dashboards for real-time monitoring:

- **Operational Dashboard**: For monitoring current grid status
- **Vulnerability Dashboard**: For tracking component risk levels
- **Policy Dashboard**: For evaluating agent recommendations

### 4. Report Generator (`report_generator.py`)

Creates structured reports with integrated visualizations:

- **Daily Summary Reports**: Overview of grid status and predictions
- **Vulnerability Assessment Reports**: Detailed component risk analysis
- **Policy Evaluation Reports**: Analysis of agent policy performance

## Integration with Other Modules

This module connects with all previous modules:

- **Module 1 (Data Management)**: Uses grid topology and weather data
- **Module 2 (Vulnerability Analysis)**: Visualizes component vulnerability scores
- **Module 3 (Failure Prediction)**: Shows prediction results and model performance
- **Module 4 (Scenario Generation)**: Visualizes scenario outcomes
- **Module 5 (Reinforcement Learning)**: Displays agent performance and policy recommendations

## Key Features

- Multiple visualization types for different aspects of grid analysis
- Interactive dashboards for real-time monitoring
- Automated report generation with customizable templates
- Comprehensive performance metrics visualization
- Integration with all previous framework components

## Architecture

The module follows a modular design with clean interfaces:

```
visualization_reporting/
├── __init__.py
├── visualization_reporting_module.py (Main interface)
├── grid_visualization.py
├── performance_visualization.py
├── dashboard.py
├── report_generator.py
├── templates/
│   └── reports/
│       ├── daily_summary.html
│       ├── vulnerability_assessment.html
│       └── policy_evaluation.html
└── assets/
    └── logo.py
```

## Example Usage

```python
from gfmf.visualization_reporting import VisualizationReportingModule

# Initialize module
viz_module = VisualizationReportingModule()

# Create vulnerability map
vulnerability_map = viz_module.create_vulnerability_map(
    map_type='network',
    include_weather=True
)

# Generate performance visualizations
performance_viz = viz_module.create_performance_visualizations(
    metrics=['accuracy', 'outage_reduction']
)

# Generate a report
report = viz_module.generate_report(
    report_type='daily_summary'
)
```

## Configuration

The module is configured via `visualization_reporting_config.yaml`, allowing customization of:

- Visualization parameters (colors, sizes, styles)
- Dashboard layout and refresh rates
- Report templates and sections
- Output formats and directories

## Test Suite

A comprehensive test suite is available in:

```
tests/visualization_reporting/test_visualization_module.py
```

## Dependencies

- Matplotlib and Seaborn for static visualizations
- NetworkX for grid topology representation
- Dash and Plotly for interactive dashboards
- Jinja2 for report template rendering
