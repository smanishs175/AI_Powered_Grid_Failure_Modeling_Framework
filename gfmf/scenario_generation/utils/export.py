#!/usr/bin/env python
"""
Export Utilities for Scenario Generation Module

This module provides functions for exporting generated scenarios to formats
compatible with the Vulnerability Analysis Module.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

def export_scenarios_to_json(scenarios: Dict[str, List[Dict[str, Any]]],
                            output_dir: str = 'data/vulnerability_analysis/input',
                            filename: str = 'scenario_set.json') -> str:
    """
    Export scenarios to a JSON file for use in the Vulnerability Analysis Module.
    
    Args:
        scenarios: Dictionary of scenarios by type
        output_dir: Directory to save exported data
        filename: Name of the output file
        
    Returns:
        Path to the exported file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a copy of the scenarios dict without validation metrics
    export_data = {}
    for k, v in scenarios.items():
        if k != 'validation_metrics' and isinstance(v, list):
            export_data[k] = v
    
    # Add metadata
    export_data['metadata'] = {
        'total_scenarios': sum(len(scenario_list) for scenario_list in export_data.values() if isinstance(scenario_list, list)),
        'scenario_types': list(export_data.keys()),
        'creation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'validation_score': scenarios.get('validation_metrics', {}).get('overall_score', None) if isinstance(scenarios.get('validation_metrics'), dict) else None
    }
    
    # Save to JSON
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Exported {export_data['metadata']['total_scenarios']} scenarios to {output_path}")
    
    return output_path

def export_scenarios_to_csv(scenarios: Dict[str, List[Dict[str, Any]]],
                           output_dir: str = 'data/vulnerability_analysis/input',
                           prefix: str = 'scenario') -> List[str]:
    """
    Export scenarios to CSV files for use in the Vulnerability Analysis Module.
    Creates multiple files:
    - scenario_metadata.csv - General information about each scenario
    - scenario_components.csv - Component failure information
    - scenario_weather.csv - Weather conditions
    - scenario_cascades.csv - Cascade propagation information
    
    Args:
        scenarios: Dictionary of scenarios by type
        output_dir: Directory to save exported data
        prefix: Prefix for output filenames
        
    Returns:
        List of paths to exported files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = []
    
    # --- 1. Create scenario metadata dataframe ---
    metadata_rows = []
    
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
            for scenario in scenario_data:
                if isinstance(scenario, dict):
                    metadata_rows.append({
                        'scenario_id': scenario.get('scenario_id', ''),
                        'scenario_type': scenario_type,
                        'severity': scenario.get('severity', 'unknown'),
                        'n_component_failures': len(scenario.get('component_failures', {})),
                        'has_cascade': len(scenario.get('cascade_propagation', {})) > 0,
                        'max_cascade_step': max([0] + [int(step) for step in scenario.get('cascade_propagation', {}).keys() 
                                                     if step != 'initial' and step.isdigit()]),
                        'total_affected_components': (
                            len(scenario.get('component_failures', {})) +
                            sum(len(failures) for step, failures in scenario.get('cascade_propagation', {}).items() 
                                if step != 'initial')
                        ),
                        'scenario_duration': scenario.get('duration', 0)
                    })
    
    # Create and save dataframe
    if metadata_rows:
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_path = os.path.join(output_dir, f"{prefix}_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        output_paths.append(metadata_path)
        logger.info(f"Exported scenario metadata to {metadata_path}")
    
    # --- 2. Create component failures dataframe ---
    component_rows = []
    
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
            for scenario in scenario_data:
                if isinstance(scenario, dict):
                    scenario_id = scenario.get('scenario_id', '')
                    
                    # Add initial failures
                    for comp_id, failure_data in scenario.get('component_failures', {}).items():
                        component_rows.append({
                            'scenario_id': scenario_id,
                            'component_id': comp_id,
                            'failure_time': failure_data.get('failure_time', 0),
                            'failure_cause': failure_data.get('failure_cause', 'unknown'),
                            'cascade_step': 'initial',
                            'failure_probability': failure_data.get('failure_probability', 0),
                            'repair_time': failure_data.get('repair_time', 0),
                            'impact_score': failure_data.get('impact_score', 0)
                        })
                    
                    # Add cascade failures
                    for step, failures in scenario.get('cascade_propagation', {}).items():
                        if step != 'initial':
                            for comp_id, cause in failures.items():
                                component_rows.append({
                                    'scenario_id': scenario_id,
                                    'component_id': comp_id,
                                    'failure_time': scenario.get('component_failures', {}).get(comp_id, {}).get('failure_time', 0) + int(step) * 0.5,
                                    'failure_cause': 'cascade',
                                    'cascade_step': step,
                                    'cascade_cause': cause,
                                    'failure_probability': 1.0,
                                    'repair_time': 0,
                                    'impact_score': 0
                                })
    
    # Create and save dataframe
    if component_rows:
        component_df = pd.DataFrame(component_rows)
        component_path = os.path.join(output_dir, f"{prefix}_components.csv")
        component_df.to_csv(component_path, index=False)
        output_paths.append(component_path)
        logger.info(f"Exported component failure data to {component_path}")
    
    # --- 3. Create weather conditions dataframe ---
    weather_rows = []
    
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
            for scenario in scenario_data:
                if isinstance(scenario, dict):
                    scenario_id = scenario.get('scenario_id', '')
                    weather = scenario.get('weather_conditions', {})
                    
                    # Process each weather factor
                    weather_row = {'scenario_id': scenario_id, 'scenario_type': scenario_type}
                    
                    for factor, value in weather.items():
                        # Handle list/tuple values
                        if isinstance(value, (list, tuple)):
                            value = sum(value) / len(value)
                        
                        weather_row[factor] = value
                    
                    weather_rows.append(weather_row)
    
    # Create and save dataframe
    if weather_rows:
        weather_df = pd.DataFrame(weather_rows)
        weather_path = os.path.join(output_dir, f"{prefix}_weather.csv")
        weather_df.to_csv(weather_path, index=False)
        output_paths.append(weather_path)
        logger.info(f"Exported weather condition data to {weather_path}")
    
    return output_paths

def export_network_data(scenarios: Dict[str, List[Dict[str, Any]]],
                       grid_components: pd.DataFrame,
                       output_dir: str = 'data/vulnerability_analysis/input',
                       filename: str = 'grid_network.pickle') -> str:
    """
    Export network representation of the grid for use in vulnerability analysis.
    
    Args:
        scenarios: Dictionary of scenarios (used to extract cascade information)
        grid_components: DataFrame with grid component information
        output_dir: Directory to save exported data
        filename: Name of the output file
        
    Returns:
        Path to the exported file
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("NetworkX library not available, can't export network data")
        return ""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a network graph
    G = nx.DiGraph()
    
    # Add nodes for all components
    for _, row in grid_components.iterrows():
        G.add_node(row['component_id'], 
                  type=row.get('type', 'unknown'),
                  voltage=row.get('voltage', 0),
                  capacity=row.get('capacity', 0),
                  age=row.get('age', 0),
                  location=row.get('location', ''),
                  vulnerability_score=row.get('vulnerability_score', 0))
    
    # Add edges based on cascade information from scenarios
    edges = set()
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
            for scenario in scenario_data:
                if isinstance(scenario, dict):
                    for step, failures in scenario.get('cascade_propagation', {}).items():
                        if step != 'initial':
                            for comp_id, cause in failures.items():
                                if cause in G and comp_id in G:
                                    edges.add((cause, comp_id))
    
    # Add edges to graph
    for source, target in edges:
        G.add_edge(source, target)
    
    # Save network
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    
    logger.info(f"Exported grid network with {len(G.nodes)} nodes and {len(G.edges)} edges to {output_path}")
    
    return output_path

def prepare_for_vulnerability_analysis(scenarios: Dict[str, List[Dict[str, Any]]],
                                     grid_components: Optional[pd.DataFrame] = None,
                                     output_dir: str = 'data/vulnerability_analysis/input',
                                     create_summary: bool = True) -> Dict[str, str]:
    """
    Prepare all necessary data for the Vulnerability Analysis Module.
    
    Args:
        scenarios: Dictionary of scenarios by type
        grid_components: DataFrame with grid component information (optional)
        output_dir: Directory to save exported data
        create_summary: Whether to create a summary file
        
    Returns:
        Dictionary with paths to exported files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    
    # 1. Export scenarios to JSON
    json_path = export_scenarios_to_json(scenarios, output_dir)
    output_paths['json'] = json_path
    
    # 2. Export scenarios to CSV
    csv_paths = export_scenarios_to_csv(scenarios, output_dir)
    output_paths['csv'] = csv_paths
    
    # 3. Export network data if grid_components is available
    if grid_components is not None:
        network_path = export_network_data(scenarios, grid_components, output_dir)
        output_paths['network'] = network_path
    
    # 4. Create summary file if requested
    if create_summary:
        # Count scenarios by type
        scenario_counts = {}
        total_components = set()
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'extreme': 0}
        
        for scenario_type, scenario_data in scenarios.items():
            if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
                scenario_counts[scenario_type] = len(scenario_data)
                
                # Count by severity
                for scenario in scenario_data:
                    if isinstance(scenario, dict):
                        severity = scenario.get('severity', 'unknown')
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                        
                        # Collect unique components
                        for comp_id in scenario.get('component_failures', {}):
                            total_components.add(comp_id)
        
        # Create summary
        summary = {
            'total_scenarios': sum(scenario_counts.values()),
            'scenario_types': scenario_counts,
            'total_components_affected': len(total_components),
            'severity_distribution': severity_counts,
            'validation_score': scenarios.get('validation_metrics', {}).get('overall_score', 'N/A'),
            'exported_files': {
                'json': os.path.basename(json_path),
                'csv': [os.path.basename(p) for p in csv_paths],
                'network': os.path.basename(output_paths.get('network', '')) if 'network' in output_paths else 'N/A'
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, 'scenario_export_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        output_paths['summary'] = summary_path
        logger.info(f"Created export summary at {summary_path}")
    
    return output_paths
