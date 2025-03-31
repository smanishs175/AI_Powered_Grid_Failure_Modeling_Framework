# Visualization and Reporting Module

This module provides visualization capabilities for the Grid Failure Modeling Framework (GFMF), including grid vulnerability visualization, performance metrics visualization, interactive dashboards, and automated report generation.

## Module Components

The module consists of the following components:

1. **Visualization and Reporting Module (`visualization_reporting_module.py`)** - The main interface for the module, integrating all visualization components.

2. **Grid Visualization (`grid_visualization.py`)** - Creates visualizations of grid topology, component vulnerability, and operational status.

3. **Performance Visualization (`performance_visualization.py`)** - Creates visualizations of model performance metrics, including ROC curves, confusion matrices, and agent learning curves.

4. **Dashboard (`dashboard.py`)** - Creates interactive dashboards for real-time monitoring of grid status, vulnerability, and policy recommendations.

5. **Report Generator (`report_generator.py`)** - Generates automated reports from visualization and analysis data.

## Key Features

- **Grid Vulnerability Visualization**
  - Network graphs showing grid topology with color-coded vulnerability
  - Heatmaps representing vulnerability scores across geographic areas
  - Geographic visualizations with weather overlays

- **Performance Metrics Visualization**
  - ROC curves and confusion matrices for prediction models
  - Learning curves for reinforcement learning agents
  - Comparative bar charts for agent performance metrics

- **Interactive Dashboards**
  - Operational dashboard for real-time grid status monitoring
  - Vulnerability dashboard for risk assessment
  - Policy dashboard for decision support

- **Automated Reports**
  - Daily summary reports
  - Vulnerability assessment reports
  - Policy evaluation reports
  - Support for HTML and PDF formats

## Integration with Other Modules

This module integrates with the other modules of the GFMF as follows:

- **Data Management Module (Module 1)**: Uses processed grid topology, weather data, and outage data for visualization.

- **Vulnerability Analysis Module (Module 2)**: Visualizes component vulnerability scores and environmental threat profiles.

- **Failure Prediction Module (Module 3)**: Visualizes failure probabilities and correlation models.

- **Scenario Generation Module (Module 4)**: Visualizes generated scenarios and cascade failure propagation.

- **Reinforcement Learning Module (Module 5)**: Visualizes agent performance metrics, learning progress, and policy recommendations.

## Usage Examples

### Creating Grid Vulnerability Visualizations

```python
from gfmf.visualization_reporting import VisualizationReportingModule

# Initialize the module
viz_module = VisualizationReportingModule('config/viz_config.yaml')

# Create a vulnerability map
vulnerability_map = viz_module.create_vulnerability_map(
    map_type='network',
    include_weather=True,
    show_predictions=True,
    output_format='png'
)

print(f"Vulnerability map saved to: {vulnerability_map['file_path']}")
```

### Creating Performance Visualizations

```python
# Create performance visualizations
performance_viz = viz_module.create_performance_visualizations(
    include_models=['failure_prediction', 'rl_agents'],
    metrics=['accuracy', 'reward', 'outage_reduction'],
    comparison_type='bar_chart'
)

for viz_type, data in performance_viz.items():
    print(f"Created {viz_type} visualizations")
```

### Launching an Interactive Dashboard

```python
# Launch an interactive dashboard
dashboard = viz_module.launch_dashboard(
    dashboard_type='operational',
    auto_refresh=True,
    refresh_interval=300  # seconds
)

print(f"Dashboard available at: {dashboard['url']}")
```

### Generating an Automated Report

```python
# Generate a report
report = viz_module.generate_report(
    report_type='daily_summary',
    include_sections=['summary', 'vulnerability', 'predictions', 'policies'],
    output_format='html'
)

print(f"Report generated at: {report['file_path']}")
```

## Configuration

The module is configured using a YAML file with the following structure:

```yaml
visualization_reporting:
  output_dir: outputs/visualization_reporting
  grid_visualization:
    node_size: 300
    font_size: 10
    color_scheme:
      operational: green
      at_risk: yellow
      failed: red
    map_style: light
    default_format: png
    dpi: 300
  performance_visualization:
    figure_size: [10, 6]
    dpi: 100
    style: whitegrid
    palette: deep
    default_format: png
  dashboard:
    port: 8050
    theme: light
    refresh_interval: 300
    default_layout: grid
    max_items_per_page: 6
  report_generator:
    template_dir: templates/reports
    default_format: pdf
    logo_path: assets/logo.png
    company_name: Grid Resilience Inc.
    default_sections: [summary, vulnerability, predictions, policies]
```

## Dependencies

This module depends on the following libraries:

- Matplotlib and Seaborn for static visualizations
- NetworkX for grid topology visualization
- Dash for interactive dashboards
- Jinja2 for report template rendering

## Testing

To test the module functionality, run the test script:

```bash
python tests/visualization_reporting/test_visualization_module.py
```
