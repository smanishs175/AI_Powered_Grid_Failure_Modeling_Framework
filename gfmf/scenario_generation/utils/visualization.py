#!/usr/bin/env python
"""
Visualization Utilities for Scenario Generation Module

This module provides visualization functions for exploring generated scenarios.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional

# Set style for plots
plt.style.use('ggplot')
sns.set_palette('viridis')

def plot_scenario_distribution(scenarios: Dict[str, List[Dict]], 
                               output_dir: str = 'outputs/scenario_visualizations',
                               filename: str = 'scenario_distribution.png') -> str:
    """
    Plot the distribution of scenario types and severities.
    
    Args:
        scenarios: Dictionary of scenarios by type
        output_dir: Directory to save visualization
        filename: Filename for the plot
        
    Returns:
        Path to saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Count scenarios by type
    scenario_counts = {}
    severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'extreme': 0}
    
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics':
            # Handle case where scenario_data is a list of scenario dictionaries
            if isinstance(scenario_data, list):
                scenario_list = scenario_data
                scenario_counts[scenario_type] = len(scenario_list)
                
                # Count by severity
                for scenario in scenario_list:
                    if isinstance(scenario, dict):
                        severity = scenario.get('severity', 'unknown')
                        if severity in severity_counts:
                            severity_counts[severity] += 1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot scenario types
    ax1.bar(scenario_counts.keys(), scenario_counts.values(), color=sns.color_palette('viridis', len(scenario_counts)))
    ax1.set_title('Scenarios by Type')
    ax1.set_ylabel('Number of Scenarios')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot severity distribution
    ax2.bar(severity_counts.keys(), severity_counts.values(), color=sns.color_palette('rocket', len(severity_counts)))
    ax2.set_title('Scenarios by Severity')
    ax2.set_ylabel('Number of Scenarios')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def plot_failure_heatmap(scenarios: Dict[str, List[Dict]], 
                        components: List[str] = None,
                        max_components: int = 20,
                        output_dir: str = 'outputs/scenario_visualizations',
                        filename: str = 'failure_heatmap.png') -> str:
    """
    Create a heatmap showing component failure frequencies across scenarios.
    
    Args:
        scenarios: Dictionary of scenarios by type
        components: List of component IDs to include (if None, use most frequent)
        max_components: Maximum number of components to show
        output_dir: Directory to save visualization
        filename: Filename for the plot
        
    Returns:
        Path to saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all scenarios
    all_scenarios = []
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
            all_scenarios.extend(scenario_data)
    
    # Count component failures
    component_failures = {}
    for scenario in all_scenarios:
        if isinstance(scenario, dict):
            for comp_id in scenario.get('component_failures', {}).keys():
                component_failures[comp_id] = component_failures.get(comp_id, 0) + 1
    
    # Select components to display
    if components is None:
        components = sorted(component_failures.items(), key=lambda x: x[1], reverse=True)
        components = [comp[0] for comp in components[:max_components]]
    
    # Create failure matrix
    scenario_types = [st for st in scenarios.keys() if st != 'validation_metrics']
    failure_matrix = np.zeros((len(scenario_types), len(components)))
    
    for i, scenario_type in enumerate(scenario_types):
        scenario_list = scenarios[scenario_type]
        if isinstance(scenario_list, list):
            for scenario in scenario_list:
                if isinstance(scenario, dict):
                    for j, comp_id in enumerate(components):
                        if comp_id in scenario.get('component_failures', {}):
                            failure_matrix[i, j] += 1
    
    # Normalize by number of scenarios
    for i, scenario_type in enumerate(scenario_types):
        num_scenarios = len(scenarios[scenario_type])
        if num_scenarios > 0:
            failure_matrix[i, :] /= num_scenarios
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define custom colormap from white to dark blue
    cmap = LinearSegmentedColormap.from_list('blue_white', ['#ffffff', '#1f77b4'])
    
    # Create heatmap
    sns.heatmap(failure_matrix, annot=True, fmt=".2f", 
                xticklabels=components, yticklabels=scenario_types,
                cmap=cmap, vmin=0, vmax=1, ax=ax)
    
    ax.set_title('Component Failure Frequency by Scenario Type')
    ax.set_xlabel('Component ID')
    ax.set_ylabel('Scenario Type')
    plt.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def visualize_cascade_network(cascading_scenario: Dict[str, Any],
                             output_dir: str = 'outputs/scenario_visualizations',
                             filename: str = 'cascade_network.png',
                             show_component_types: bool = True,
                             detailed_labels: bool = False) -> str:
    """
    Visualize the cascading failure network for a specific scenario.
    
    Args:
        cascading_scenario: A scenario with cascade information
        output_dir: Directory to save visualization
        filename: Filename for the plot
        show_component_types: Whether to color nodes by component type
        detailed_labels: Whether to show detailed node labels including load/capacity
        
    Returns:
        Path to saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for enhanced cascade data structure
    if 'cascade' in cascading_scenario and 'network_data' in cascading_scenario['cascade']:
        # Use enhanced data structure from new model
        return _visualize_enhanced_cascade(cascading_scenario, output_dir, filename, show_component_types, detailed_labels)
    
    # Fallback to basic visualization for older cascade data structure
    # Create network graph
    G = nx.DiGraph()
    
    # Add nodes for all components
    initial_failures = cascading_scenario.get('component_failures', {})
    cascade_progression = cascading_scenario.get('cascade', {}).get('cascade_progression', {})
    
    # Old style cascade propagation
    cascade_propagation = cascading_scenario.get('cascade_propagation', {}) 
    
    # Add initial failures as red nodes
    for comp_id in initial_failures:
        G.add_node(comp_id, color='red', initial=True)
    
    # Add cascade failures and edges
    if cascade_progression:
        # New format - cascade progression from steps
        for step_data in cascade_progression[1:]:  # Skip initial step
            if 'failure_details' in step_data:
                for failure in step_data['failure_details']:
                    comp_id = failure['component_id']
                    cause_id = failure.get('cause_id', '')
                    
                    if comp_id not in G:
                        G.add_node(comp_id, color='orange', initial=False)
                    
                    # Add edge from cause to component if both exist
                    if cause_id and cause_id in G:
                        G.add_edge(cause_id, comp_id)
            else:
                # Simple handling without detailed failure info
                for comp_id in step_data.get('newly_failed', []):
                    if comp_id not in G:
                        G.add_node(comp_id, color='orange', initial=False)
    elif cascade_propagation:
        # Old format - cascade propagation by step
        for step, failures in cascade_propagation.items():
            if step == 'initial':
                continue
                
            for comp_id, cause in failures.items():
                if comp_id not in G:
                    G.add_node(comp_id, color='orange', initial=False)
                
                # Add edge from cause to component
                if cause in G:
                    G.add_edge(cause, comp_id)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define node colors
    node_colors = [data.get('color', 'blue') for node, data in G.nodes(data=True)]
    
    # Define node sizes - larger for initial failures
    node_sizes = [300 if data.get('initial', False) else 150 
                 for node, data in G.nodes(data=True)]
    
    # Create layout - force-directed for smaller graphs, shell for larger
    if len(G) < 20:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.shell_layout(G)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, width=1.5, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    ax.set_title(f"Cascading Failure Network - Scenario {cascading_scenario.get('scenario_id', 'Unknown')}")
    ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Initial Failure'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Cascade Failure')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _visualize_enhanced_cascade(cascading_scenario: Dict[str, Any],
                             output_dir: str = 'outputs/scenario_visualizations',
                             filename: str = 'cascade_network.png',
                             show_component_types: bool = True,
                             detailed_labels: bool = False) -> str:
    """
    Visualize the enhanced cascading failure network with detailed data.
    
    This function uses the enhanced data structure from the improved cascade model.
    
    Args:
        cascading_scenario: A scenario with enhanced cascade information
        output_dir: Directory to save visualization
        filename: Filename for the plot
        show_component_types: Whether to color nodes by component type
        detailed_labels: Whether to show detailed node labels
        
    Returns:
        Path to saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract cascade data
    cascade_data = cascading_scenario['cascade']
    network_data = cascade_data.get('network_data', {})
    failure_chains = cascade_data.get('failure_chains', [])
    
    # Create network graph
    G = nx.DiGraph()
    
    # Component type colors
    type_colors = {
        'generator': '#e41a1c',    # Red
        'plant': '#e41a1c',        # Red
        'substation': '#377eb8',   # Blue
        'transformer': '#4daf4a',  # Green
        'transmission': '#984ea3', # Purple
        'line': '#ff7f00',         # Orange
        'distribution': '#ffff33',  # Yellow
        'unknown': '#999999'       # Gray
    }
    
    # Add nodes from network data
    if 'nodes' in network_data:
        for node in network_data['nodes']:
            node_id = node['id']
            node_type = node.get('type', 'unknown').lower()
            failed = node.get('failed', False)
            failure_step = node.get('failure_step', -1)
            
            # Default color based on component type
            color = '#999999'  # Default gray
            for type_key, type_color in type_colors.items():
                if type_key in node_type:
                    color = type_color
                    break
            
            # Adjustments for failed components
            if not show_component_types and failed:
                if failure_step == 0:
                    color = '#d62728'  # Initial failures - bright red
                else:
                    # Color gradient from orange to yellow based on step
                    color = f'#{min(255, 200 + failure_step * 20):02x}{100 + failure_step * 20:02x}00'
            
            # Node properties
            G.add_node(
                node_id,
                type=node_type,
                failed=failed,
                failure_step=failure_step,
                color=color,
                load=node.get('load', 0),
                capacity=node.get('capacity', 100),
                vulnerability=node.get('vulnerability', 0.5)
            )
    
    # Add edges from network data
    if 'edges' in network_data:
        for edge in network_data['edges']:
            source = edge['source']
            target = edge['target']
            is_cascade_path = edge.get('is_cascade_path', False)
            
            G.add_edge(
                source, 
                target, 
                weight=edge.get('weight', 1.0),
                is_cascade_path=is_cascade_path
            )
    
    # If no edges or nodes yet, try using failure chains
    if len(G.nodes) == 0 and len(failure_chains) > 0:
        # Add nodes and edges from failure chains
        for chain_data in failure_chains:
            chain = chain_data.get('chain', [])
            for i, node_id in enumerate(chain):
                # Add node if it doesn't exist
                if node_id not in G:
                    G.add_node(
                        node_id,
                        failed=True,
                        failure_step=i,
                        color='#ff9900' if i > 0 else '#ff0000',  # Orange for cascade, red for initial
                        type='unknown'
                    )
                
                # Add edge to next node in chain
                if i < len(chain) - 1:
                    G.add_edge(node_id, chain[i+1], is_cascade_path=True)
    
    # If still no data, return error message
    if len(G.nodes) == 0:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "No cascade data available for visualization", 
                 horizontalalignment='center', fontsize=14, color='red')
        plt.axis('off')
        
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define node properties
    node_colors = [data.get('color', '#999999') for _, data in G.nodes(data=True)]
    
    # Scale node sizes based on capacity if available, otherwise use default sizes
    node_sizes = []
    for _, data in G.nodes(data=True):
        if 'capacity' in data and data['capacity'] > 0:
            # Scale node size based on capacity (with minimum and maximum size)
            size = max(100, min(1000, data['capacity'] * 5))
        else:
            # Default sizes based on whether the node failed
            size = 300 if data.get('failed', False) else 150
        node_sizes.append(size)
    
    # Determine layout based on graph size and structure
    if len(G) < 20:
        pos = nx.spring_layout(G, seed=42, k=0.5)  # k controls spacing
    elif len(G) < 50:
        pos = nx.kamada_kawai_layout(G)  # Better for medium-sized graphs
    else:
        # For large graphs, use a faster layout algorithm
        pos = nx.spring_layout(G, seed=42, iterations=50) 
    
    # Draw regular edges first
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('is_cascade_path', False)]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='#cccccc', 
                          arrows=True, arrowsize=10, width=1.0, alpha=0.5)
    
    # Draw cascade path edges with different style
    cascade_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_cascade_path', False)]
    nx.draw_networkx_edges(G, pos, edgelist=cascade_edges, edge_color='#ff0000', 
                          arrows=True, arrowsize=15, width=2.0, alpha=0.8)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Create node labels based on detail level
    if detailed_labels:
        labels = {}
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            if 'load' in data and 'capacity' in data:
                load_ratio = data['load'] / data['capacity'] if data['capacity'] > 0 else 0
                labels[node] = f"{node}\n{node_type[:10]}\n{load_ratio:.1f} load ratio"
            else:
                labels[node] = f"{node}\n{node_type[:10]}"
    else:
        # Simple labels - just node ID
        labels = {node: str(node) for node in G.nodes()}
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', font_weight='bold')
    
    # Title
    total_steps = cascade_data.get('total_steps', 0)
    failed_count = cascade_data.get('final_failed_count', 0)
    ax.set_title(f"Cascading Failure Network - Scenario {cascading_scenario.get('scenario_id', 'Unknown')}\n"
                f"{total_steps} steps, {failed_count} failed components", fontsize=14)
    ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = []
    
    # Add component type legend if showing component types
    if show_component_types:
        for component_type, color in type_colors.items():
            if any(component_type in data.get('type', '').lower() for _, data in G.nodes(data=True)):
                legend_elements.append(Patch(facecolor=color, edgecolor='black', label=component_type.capitalize()))
    else:
        # Add failure type legend
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Initial Failure'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9900', markersize=10, label='Cascade Failure')
        ])
    
    # Add edge type legend
    legend_elements.extend([
        Line2D([0], [0], color='#ff0000', lw=2, label='Cascade Path'),
        Line2D([0], [0], color='#cccccc', lw=1, label='Grid Connection')
    ])
    
    # Place legend
    if len(legend_elements) > 0:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create an animated version showing cascade propagation if enough steps
    if len(cascade_data.get('cascade_progression', [])) > 2:
        _create_cascade_animation(cascading_scenario, output_dir, filename.replace('.png', '_animated.gif'))
    
    return output_path


def _create_cascade_animation(cascading_scenario: Dict[str, Any],
                             output_dir: str,
                             filename: str) -> str:
    """
    Create an animated visualization of cascade progression.
    
    Args:
        cascading_scenario: Scenario with cascade data
        output_dir: Directory to save the animation
        filename: Filename for the animation
        
    Returns:
        Path to saved animation file
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        return ""  # Animation libraries not available
        
    # Get cascade data
    cascade_data = cascading_scenario['cascade']
    progression = cascade_data.get('cascade_progression', [])
    network_data = cascade_data.get('network_data', {})
    
    if len(progression) <= 1 or not network_data:
        return ""  # Not enough data for animation
    
    # Create the base graph
    G = nx.DiGraph()
    pos = {}  # Store positions
    
    # Add all nodes and edges from network data
    for node in network_data.get('nodes', []):
        G.add_node(node['id'], **node)
    
    for edge in network_data.get('edges', []):
        G.add_edge(edge['source'], edge['target'], **edge)
        
    # Create layout (will be fixed throughout animation)
    if len(G) < 30:
        pos = nx.spring_layout(G, seed=42, k=0.5)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Animation update function
    def update(frame):
        ax.clear()
        ax.axis('off')
        
        # Set title with frame info
        if frame == 0:
            ax.set_title("Initial State - No Failures", fontsize=14)
        elif frame <= len(progression):
            # Make sure we have data for this frame
            step_data = progression[frame-1]  # Adjust index since frame 0 is initial state
            step_num = step_data.get('step', frame)
            newly_failed = step_data.get('newly_failed', [])
            all_failed = step_data.get('all_failed', [])
            ax.set_title(f"Step {step_num}: {len(newly_failed)} new failures, {len(all_failed)} total", fontsize=14)
        else:
            # Safety for unexpected frame number
            ax.set_title(f"Step {frame}", fontsize=14)
        
        # Determine node colors based on current frame
        node_colors = {}
        node_sizes = {}
        
        for node in G.nodes():
            # Default color: gray
            node_colors[node] = '#cccccc'
            node_sizes[node] = 100
            
            # Check if this node has failed by this frame
            for i in range(min(frame, len(progression))):
                step_data = progression[i]
                if node in step_data.get('newly_failed', []):
                    # Node has failed - color based on step
                    if i == 0:  # Initial failure
                        node_colors[node] = '#d62728'  # Red
                    else:  # Cascade failure
                        # Color from orange to yellow based on step 
                        node_colors[node] = f'#{min(255, 200 + i * 10):02x}{100 + i * 15:02x}00'
                    
                    node_sizes[node] = 200
                    break
        
        # Draw network components
        # First draw all edges
        nx.draw_networkx_edges(G, pos, edge_color='#cccccc', alpha=0.3, width=1.0)
        
        # Draw all nodes with their current colors
        for node, color in node_colors.items():
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=[color], 
                                   alpha=0.8, node_size=node_sizes[node])
        
        # Add labels - smaller font for better visibility
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        # If this is a cascade step, highlight new failures and connections
        if frame > 0 and frame <= len(progression):
            # Get step data (safely)
            step_data = progression[frame-1]  # Adjust index since frame 0 is initial state
            newly_failed = step_data.get('newly_failed', [])
            
            # Highlight the newly failed nodes
            nx.draw_networkx_nodes(G, pos, nodelist=newly_failed, 
                                node_color='#ff9900', alpha=1.0, 
                                node_size=[node_sizes[n] * 1.5 for n in newly_failed])
                                
            # Highlight the edges that caused cascade in this step
            cascade_edges = []
            for failure in step_data.get('failure_details', []):
                if 'cause_id' in failure and failure['cause_id'] in G:
                    source = failure['cause_id']
                    target = failure['component_id']
                    if G.has_edge(source, target):
                        cascade_edges.append((source, target))
            
            nx.draw_networkx_edges(G, pos, edgelist=cascade_edges, edge_color='#ff0000', 
                                  alpha=1.0, width=2.0, arrows=True, arrowsize=15)
    
    # Create animation - only if we have enough frames
    if len(progression) >= 1:
        frames = len(progression) + 1  # +1 for initial state with no failures
        anim = FuncAnimation(fig, update, frames=frames, repeat=True, interval=1500)
        
        try:
            # Save animation
            output_path = os.path.join(output_dir, filename)
            writer = PillowWriter(fps=1)
            anim.save(output_path, writer=writer)
            plt.close(fig)
            return output_path
        except Exception as e:
            plt.close(fig)
            return ""  # Error during animation
    else:
        plt.close(fig)
        return ""  # Not enough data for animation
    
    return output_path

def plot_weather_conditions(scenarios: Dict[str, List[Dict]],
                          output_dir: str = 'outputs/scenario_visualizations',
                          filename: str = 'weather_conditions.png') -> str:
    """
    Plot the distribution of weather conditions across scenario types.
    
    Args:
        scenarios: Dictionary of scenarios by type
        output_dir: Directory to save visualization
        filename: Filename for the plot
        
    Returns:
        Path to saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect weather data by scenario type
    weather_data = {}
    weather_factors = ['temperature', 'wind_speed', 'precipitation', 'humidity']
    
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
            weather_data[scenario_type] = {factor: [] for factor in weather_factors}
            
            for scenario in scenario_data:
                if isinstance(scenario, dict):
                    weather = scenario.get('weather_conditions', {})
                    
                    for factor in weather_factors:
                        value = weather.get(factor, None)
                        
                        # Handle list/tuple values
                        if isinstance(value, (list, tuple)):
                            value = sum(value) / len(value)
                            
                        if value is not None:
                            weather_data[scenario_type][factor].append(value)
    
    # Create figure with subplots for each weather factor
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot distributions for each weather factor
    for i, factor in enumerate(weather_factors):
        ax = axes[i]
        
        for scenario_type, data in weather_data.items():
            if data[factor]:
                sns.kdeplot(data[factor], ax=ax, label=scenario_type)
        
        ax.set_title(f'{factor.capitalize()} Distribution')
        ax.set_xlabel(factor)
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def create_scenario_dashboard(scenarios: Dict[str, List[Dict]], 
                             components_df: Optional[pd.DataFrame] = None,
                             output_dir: str = 'outputs/scenario_visualizations') -> List[str]:
    """
    Create a comprehensive dashboard of visualizations for the scenario set.
    
    Args:
        scenarios: Dictionary of scenarios by type
        components_df: DataFrame with component information (optional)
        output_dir: Directory to save visualization
        
    Returns:
        List of paths to saved visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all visualizations
    outputs = []
    
    # 1. Scenario distribution
    dist_path = plot_scenario_distribution(scenarios, output_dir)
    outputs.append(dist_path)
    
    # 2. Failure heatmap
    heatmap_path = plot_failure_heatmap(scenarios, output_dir=output_dir)
    outputs.append(heatmap_path)
    
    # 3. Weather conditions
    weather_path = plot_weather_conditions(scenarios, output_dir=output_dir)
    outputs.append(weather_path)
    
    # 4. Cascade network for a high-severity scenario
    # Find a scenario with cascades
    cascade_scenario = None
    for scenario_type, scenario_data in scenarios.items():
        if scenario_type != 'validation_metrics' and isinstance(scenario_data, list):
            for scenario in scenario_data:
                if isinstance(scenario, dict) and scenario.get('severity', '') == 'high' and scenario.get('cascade_propagation'):
                    cascade_scenario = scenario
                    break
            if cascade_scenario:
                break
    
    if cascade_scenario:
        cascade_path = visualize_cascade_network(cascade_scenario, output_dir, 
                                               f'cascade_network_{cascade_scenario.get("scenario_id", "unknown")}.png')
        outputs.append(cascade_path)
    
    # 5. Generate summary HTML
    html_content = f"""
    <html>
    <head>
        <title>Scenario Generation Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333366; }}
            .visual-container {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
            .metrics {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Scenario Generation Dashboard</h1>
        
        <div class="metrics">
            <h2>Validation Metrics</h2>
            <ul>
    """
    
    # Add validation metrics if available
    if 'validation_metrics' in scenarios:
        for metric, value in scenarios['validation_metrics'].items():
            if isinstance(value, (int, float)):
                html_content += f"<li><strong>{metric}:</strong> {value:.2f}</li>\n"
            else:
                html_content += f"<li><strong>{metric}:</strong> {value}</li>\n"
    
    html_content += """
            </ul>
        </div>
        
        <div class="visual-container">
            <h2>Scenario Distribution</h2>
            <img src="scenario_distribution.png" alt="Scenario Distribution">
        </div>
        
        <div class="visual-container">
            <h2>Component Failure Heatmap</h2>
            <img src="failure_heatmap.png" alt="Component Failure Heatmap">
        </div>
        
        <div class="visual-container">
            <h2>Weather Condition Distribution</h2>
            <img src="weather_conditions.png" alt="Weather Condition Distribution">
        </div>
    """
    
    if cascade_scenario:
        cascade_filename = f'cascade_network_{cascade_scenario.get("scenario_id", "unknown")}.png'
        html_content += f"""
        <div class="visual-container">
            <h2>Sample Cascading Failure Network</h2>
            <img src="{cascade_filename}" alt="Cascading Failure Network">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save HTML file
    html_path = os.path.join(output_dir, 'scenario_dashboard.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    outputs.append(html_path)
    
    return outputs
