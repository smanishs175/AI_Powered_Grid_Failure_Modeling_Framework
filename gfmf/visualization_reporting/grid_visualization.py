"""
Grid visualization component for the Grid Failure Modeling Framework.

This module provides classes and functions for visualizing grid topology,
vulnerability, and operational status.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime


class GridVisualization:
    """
    Class for creating grid topology and vulnerability visualizations.
    
    This class provides methods for creating different types of grid visualizations,
    including network graphs, heatmaps, and geographic visualizations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the GridVisualization class.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Grid Visualization component")
        
        # Set default configuration if not provided
        self.config = config or {}
        
        # Set default visualization parameters
        self.node_size = self.config.get('node_size', 300)
        self.font_size = self.config.get('font_size', 10)
        self.dpi = self.config.get('dpi', 300)
        
        # Set color scheme
        color_scheme = self.config.get('color_scheme', {})
        self.color_operational = color_scheme.get('operational', 'green')
        self.color_at_risk = color_scheme.get('at_risk', 'yellow')
        self.color_failed = color_scheme.get('failed', 'red')
        
        # Initialize figure style
        self.map_style = self.config.get('map_style', 'default')
        # Use a valid matplotlib style
        try:
            plt.style.use(self.map_style)
        except:
            # Fallback to default style if specified style is not available
            self.logger.warning(f"Style '{self.map_style}' not available, using default style")
            plt.style.use('default')
    
    def create_vulnerability_map(self, map_type='network', include_weather=True, 
                                 show_predictions=True, output_format='png', output_path=None):
        """
        Create a grid vulnerability map visualization.
        
        Args:
            map_type (str): Type of map ('network', 'heatmap', 'geographic').
            include_weather (bool): Whether to include weather overlays.
            show_predictions (bool): Whether to show failure predictions.
            output_format (str): Output format ('png', 'svg', 'interactive').
            output_path (str, optional): Path where to save the visualization.
            
        Returns:
            dict: Dictionary with visualization metadata and paths.
        """
        # Generate a default output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vulnerability_map_{map_type}_{timestamp}.{output_format}"
            output_path = os.path.join('outputs/visualization_reporting/grid_visualizations', filename)
        
        # Get grid data
        grid_data = self._load_grid_data()
        
        # Get vulnerability data if showing predictions
        vulnerability_data = None
        if show_predictions:
            vulnerability_data = self._load_vulnerability_data()
        
        # Get weather data if including weather
        weather_data = None
        if include_weather:
            weather_data = self._load_weather_data()
        
        # Create the appropriate visualization based on map_type
        if map_type == 'network':
            fig = self._create_network_visualization(grid_data, vulnerability_data, weather_data)
        elif map_type == 'heatmap':
            fig = self._create_heatmap_visualization(grid_data, vulnerability_data, weather_data)
        elif map_type == 'geographic':
            fig = self._create_geographic_visualization(grid_data, vulnerability_data, weather_data)
        else:
            self.logger.error(f"Unknown map type: {map_type}")
            raise ValueError(f"Unknown map type: {map_type}")
        
        # Save the figure
        if output_format != 'interactive':
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Prepare result metadata
        result = {
            'file_path': output_path,
            'map_type': map_type,
            'includes_weather': include_weather,
            'includes_predictions': show_predictions,
            'format': output_format,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def get_vulnerability_data(self):
        """
        Get vulnerability data for visualization.
        
        Returns:
            dict: Dictionary with vulnerability data.
        """
        return self._load_vulnerability_data()
    
    def plot_vulnerability_map(self, vulnerability_data, ax=None):
        """
        Plot vulnerability map on a given axis.
        
        Args:
            vulnerability_data (dict): Vulnerability data.
            ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
            
        Returns:
            matplotlib.axes.Axes: The axis with the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        grid_data = self._load_grid_data()
        self._plot_network_graph(grid_data, vulnerability_data, ax)
        
        return ax
    
    def _load_grid_data(self):
        """
        Load grid topology data.
        
        Returns:
            dict: Dictionary with grid topology data.
        """
        # In a real implementation, this would load data from Module 1
        # For now, we'll create mock data
        try:
            # Try to load from Module 1 output
            grid_filepath = 'data/data_management/processed/grid_topology.json'
            if os.path.exists(grid_filepath):
                import json
                with open(grid_filepath, 'r') as f:
                    grid_data = json.load(f)
                self.logger.info("Loaded grid data from Module 1 output")
                return grid_data
        except Exception as e:
            self.logger.warning(f"Could not load grid data from Module 1: {e}")
        
        # If loading fails, generate mock data
        self.logger.info("Generating mock grid data")
        nodes = {
            f"n{i}": {
                "id": f"n{i}",
                "type": "generator" if i % 5 == 0 else "load" if i % 5 == 1 else "substation",
                "capacity": 100 * (i % 5 + 1),
                "voltage": 230,
                "age": i % 20,
                "x": np.random.uniform(0, 100),
                "y": np.random.uniform(0, 100)
            }
            for i in range(1, 31)
        }
        
        lines = {}
        for i in range(1, 40):
            from_node = f"n{np.random.randint(1, 31)}"
            to_node = f"n{np.random.randint(1, 31)}"
            while to_node == from_node:
                to_node = f"n{np.random.randint(1, 31)}"
            
            lines[f"l{i}"] = {
                "id": f"l{i}",
                "from": from_node,
                "to": to_node,
                "capacity": 150 * (i % 3 + 1),
                "length": np.random.uniform(10, 50),
                "voltage": 230,
                "age": i % 15,
                "material": "aluminum" if i % 2 == 0 else "copper"
            }
        
        return {"nodes": nodes, "lines": lines}
    
    def _load_vulnerability_data(self):
        """
        Load vulnerability data for visualization.
        
        Returns:
            dict: Dictionary with vulnerability scores for components.
        """
        # In a real implementation, this would load data from Module 2 and 3
        # For now, we'll create mock data
        try:
            # Try to load from Module 2 output
            vulnerability_filepath = 'data/vulnerability_analysis/component_scores.npy'
            if os.path.exists(vulnerability_filepath):
                vulnerability_data = np.load(vulnerability_filepath, allow_pickle=True).item()
                self.logger.info("Loaded vulnerability data from Module 2 output")
                return vulnerability_data
        except Exception as e:
            self.logger.warning(f"Could not load vulnerability data from Module 2: {e}")
        
        # If loading fails, generate mock data
        self.logger.info("Generating mock vulnerability data")
        
        grid_data = self._load_grid_data()
        node_ids = list(grid_data['nodes'].keys())
        line_ids = list(grid_data['lines'].keys())
        
        vulnerability_data = {
            'nodes': {node_id: np.random.uniform(0, 1) for node_id in node_ids},
            'lines': {line_id: np.random.uniform(0, 1) for line_id in line_ids}
        }
        
        return vulnerability_data
    
    def _load_weather_data(self):
        """
        Load weather data for visualization overlays.
        
        Returns:
            dict: Dictionary with weather data.
        """
        # In a real implementation, this would load data from Module 1
        # For now, we'll create mock data
        try:
            # Try to load from Module 1 output
            weather_filepath = 'data/data_management/processed/weather_data.csv'
            if os.path.exists(weather_filepath):
                weather_data = pd.read_csv(weather_filepath)
                self.logger.info("Loaded weather data from Module 1 output")
                
                # Process into a suitable format for visualization
                return {
                    'temperature': weather_data['temperature'].tolist(),
                    'wind_speed': weather_data['wind_speed'].tolist(),
                    'precipitation': weather_data['precipitation'].tolist(),
                    'timestamps': weather_data['timestamp'].tolist()
                }
        except Exception as e:
            self.logger.warning(f"Could not load weather data from Module 1: {e}")
        
        # If loading fails, generate mock data
        self.logger.info("Generating mock weather data")
        
        # Generate grid points for weather visualization
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x, y)
        
        # Generate weather fields
        temperature = 20 + 5 * np.sin(X / 10) * np.cos(Y / 10)
        wind_speed = 5 + 3 * np.cos(X / 15) * np.sin(Y / 15)
        precipitation = np.exp(-((X - 50) ** 2 + (Y - 50) ** 2) / 1000) * 50
        
        return {
            'x': X,
            'y': Y,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        }
    
    def _create_network_visualization(self, grid_data, vulnerability_data=None, weather_data=None):
        """
        Create a network graph visualization of the grid.
        
        Args:
            grid_data (dict): Grid topology data.
            vulnerability_data (dict, optional): Vulnerability scores.
            weather_data (dict, optional): Weather data.
            
        Returns:
            matplotlib.figure.Figure: The figure with the visualization.
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node_id, node_data in grid_data['nodes'].items():
            pos = (node_data.get('x', 0), node_data.get('y', 0))
            G.add_node(node_id, pos=pos, **node_data)
        
        # Add edges
        for line_id, line_data in grid_data['lines'].items():
            from_node = line_data['from']
            to_node = line_data['to']
            G.add_edge(from_node, to_node, **line_data)
        
        # Set node positions for layout
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        
        # Determine node colors based on vulnerability or type
        node_colors = []
        for node in G.nodes():
            if vulnerability_data:
                score = vulnerability_data['nodes'].get(node, 0)
                if score > 0.7:
                    color = self.color_failed
                elif score > 0.3:
                    color = self.color_at_risk
                else:
                    color = self.color_operational
            else:
                node_type = G.nodes[node].get('type', 'unknown')
                if node_type == 'generator':
                    color = 'blue'
                elif node_type == 'load':
                    color = 'red'
                else:
                    color = 'gray'
            node_colors.append(color)
        
        # Determine node sizes based on capacity
        node_sizes = []
        for node in G.nodes():
            capacity = G.nodes[node].get('capacity', 100)
            node_sizes.append(self.node_size * (capacity / 100))
        
        # Determine edge widths based on capacity and color based on vulnerability
        edge_widths = []
        edge_colors = []
        for u, v in G.edges():
            edge_data = G.get_edge_data(u, v)
            capacity = edge_data.get('capacity', 100)
            edge_widths.append(1 + (capacity / 50))
            
            if vulnerability_data:
                line_id = edge_data.get('id', f"{u}_{v}")
                score = vulnerability_data['lines'].get(line_id, 0)
                if score > 0.7:
                    color = self.color_failed
                elif score > 0.3:
                    color = self.color_at_risk
                else:
                    color = self.color_operational
            else:
                color = 'black'
            edge_colors.append(color)
        
        # Draw network
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        
        # Add labels
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=self.font_size)
        
        # Add weather overlay if provided
        if weather_data and isinstance(weather_data, dict) and 'temperature' in weather_data:
            if 'x' in weather_data and 'y' in weather_data:
                # Add temperature contour
                contour = ax.contourf(
                    weather_data['x'], weather_data['y'], weather_data['temperature'],
                    alpha=0.3, cmap='coolwarm', levels=15
                )
                plt.colorbar(contour, ax=ax, label='Temperature (°C)')
        
        # Set title and labels
        plt.title('Grid Vulnerability Network Visualization', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def _create_heatmap_visualization(self, grid_data, vulnerability_data=None, weather_data=None):
        """
        Create a heatmap visualization of grid vulnerability.
        
        Args:
            grid_data (dict): Grid topology data.
            vulnerability_data (dict, optional): Vulnerability scores.
            weather_data (dict, optional): Weather data.
            
        Returns:
            matplotlib.figure.Figure: The figure with the visualization.
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a 2D grid
        grid_size = 100
        vulnerability_grid = np.zeros((grid_size, grid_size))
        
        # If we have vulnerability data, populate the grid
        if vulnerability_data:
            # Populate grid with node vulnerabilities
            for node_id, node_data in grid_data['nodes'].items():
                x = int(node_data.get('x', 0))
                y = int(node_data.get('y', 0))
                
                # Keep within bounds
                x = max(0, min(x, grid_size - 1))
                y = max(0, min(y, grid_size - 1))
                
                score = vulnerability_data['nodes'].get(node_id, 0)
                vulnerability_grid[y, x] = max(vulnerability_grid[y, x], score)
            
            # Interpolate to fill in gaps - simple diffusion approach
            for _ in range(10):
                temp_grid = vulnerability_grid.copy()
                for i in range(1, grid_size - 1):
                    for j in range(1, grid_size - 1):
                        if vulnerability_grid[i, j] == 0:
                            # Simple average of neighbors
                            temp_grid[i, j] = (
                                vulnerability_grid[i-1, j] + 
                                vulnerability_grid[i+1, j] + 
                                vulnerability_grid[i, j-1] + 
                                vulnerability_grid[i, j+1]
                            ) / 4
                vulnerability_grid = temp_grid
        
        # Create a custom colormap for vulnerability
        colors = [(0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1)]  # Green, Yellow, Red with alpha
        cmap_name = 'vulnerability_cmap'
        vulnerability_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        
        # Plot the heatmap
        im = ax.imshow(vulnerability_grid, cmap=vulnerability_cmap, origin='lower', 
                       extent=[0, grid_size, 0, grid_size])
        plt.colorbar(im, ax=ax, label='Vulnerability Score')
        
        # Add weather overlay if provided
        if weather_data and isinstance(weather_data, dict) and 'temperature' in weather_data:
            if 'x' in weather_data and 'y' in weather_data:
                # Add weather as contour lines
                contour = ax.contour(
                    weather_data['x'], weather_data['y'], weather_data['temperature'],
                    colors='blue', alpha=0.7, levels=10
                )
                plt.clabel(contour, inline=1, fontsize=8)
        
        # Add node markers
        for node_id, node_data in grid_data['nodes'].items():
            x = node_data.get('x', 0)
            y = node_data.get('y', 0)
            node_type = node_data.get('type', 'unknown')
            
            marker = 's' if node_type == 'generator' else 'o' if node_type == 'load' else '^'
            color = 'blue' if node_type == 'generator' else 'red' if node_type == 'load' else 'white'
            
            ax.plot(x, y, marker=marker, markersize=10, markeredgecolor='black', 
                    markerfacecolor=color, alpha=0.9)
            ax.text(x, y, node_id, fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))
        
        # Set title and labels
        plt.title('Grid Vulnerability Heatmap', fontsize=14)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.tight_layout()
        
        return fig
    
    def _create_geographic_visualization(self, grid_data, vulnerability_data=None, weather_data=None):
        """
        Create a geographic visualization of the grid.
        
        Args:
            grid_data (dict): Grid topology data.
            vulnerability_data (dict, optional): Vulnerability scores.
            weather_data (dict, optional): Weather data.
            
        Returns:
            matplotlib.figure.Figure: The figure with the visualization.
        """
        # In a real implementation, this would use real geographic data
        # For this mock implementation, we'll create a simplified map view
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create background map (simplified)
        ax.set_facecolor('#e0f3f8')  # Light blue background for water
        
        # Add some random land masses
        land_polygons = [
            np.array([[20, 20], [80, 20], [80, 40], [60, 60], [40, 70], [20, 50]]),
            np.array([[70, 70], [90, 70], [90, 90], [80, 95], [70, 90]])
        ]
        
        for polygon in land_polygons:
            ax.fill(polygon[:, 0], polygon[:, 1], color='#a1d99b', alpha=0.7)  # Light green for land
        
        # Create NetworkX graph for grid components
        G = nx.Graph()
        
        # Add nodes
        for node_id, node_data in grid_data['nodes'].items():
            pos = (node_data.get('x', 0), node_data.get('y', 0))
            G.add_node(node_id, pos=pos, **node_data)
        
        # Add edges
        for line_id, line_data in grid_data['lines'].items():
            from_node = line_data['from']
            to_node = line_data['to']
            G.add_edge(from_node, to_node, **line_data)
        
        # Set node positions for layout
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        
        # Determine node colors based on vulnerability or type
        node_colors = []
        for node in G.nodes():
            if vulnerability_data:
                score = vulnerability_data['nodes'].get(node, 0)
                if score > 0.7:
                    color = self.color_failed
                elif score > 0.3:
                    color = self.color_at_risk
                else:
                    color = self.color_operational
            else:
                node_type = G.nodes[node].get('type', 'unknown')
                if node_type == 'generator':
                    color = 'blue'
                elif node_type == 'load':
                    color = 'red'
                else:
                    color = 'gray'
            node_colors.append(color)
        
        # Determine node sizes based on capacity
        node_sizes = []
        for node in G.nodes():
            capacity = G.nodes[node].get('capacity', 100)
            node_sizes.append(self.node_size * (capacity / 100))
        
        # Determine edge widths based on capacity and color based on vulnerability
        edge_widths = []
        edge_colors = []
        for u, v in G.edges():
            edge_data = G.get_edge_data(u, v)
            capacity = edge_data.get('capacity', 100)
            edge_widths.append(1 + (capacity / 50))
            
            if vulnerability_data:
                line_id = edge_data.get('id', f"{u}_{v}")
                score = vulnerability_data['lines'].get(line_id, 0)
                if score > 0.7:
                    color = self.color_failed
                elif score > 0.3:
                    color = self.color_at_risk
                else:
                    color = self.color_operational
            else:
                color = 'black'
            edge_colors.append(color)
        
        # Draw network on map
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        
        # Add labels with smaller font and background
        labels = {node: node for node in G.nodes()}
        for node, (x, y) in pos.items():
            ax.text(x, y, node, fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))
        
        # Add weather overlay if provided
        if weather_data and isinstance(weather_data, dict) and 'precipitation' in weather_data:
            if 'x' in weather_data and 'y' in weather_data:
                # Add precipitation contour
                contour = ax.contourf(
                    weather_data['x'], weather_data['y'], weather_data['precipitation'],
                    alpha=0.4, cmap='Blues', levels=10
                )
                plt.colorbar(contour, ax=ax, label='Precipitation (mm)')
        
        # Set title and labels
        plt.title('Geographic Grid Vulnerability Visualization', fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Set axis limits
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        
        plt.tight_layout()
        
        return fig
    
    def _plot_network_graph(self, grid_data, vulnerability_data, ax):
        """
        Plot a network graph on a given axis.
        
        Args:
            grid_data (dict): Grid topology data.
            vulnerability_data (dict): Vulnerability data.
            ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
        """
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node_id, node_data in grid_data['nodes'].items():
            pos = (node_data.get('x', 0), node_data.get('y', 0))
            G.add_node(node_id, pos=pos, **node_data)
        
        # Add edges
        for line_id, line_data in grid_data['lines'].items():
            from_node = line_data['from']
            to_node = line_data['to']
            G.add_edge(from_node, to_node, **line_data)
        
        # Set node positions for layout
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        
        # Determine node colors and sizes
        node_colors = [
            self.color_failed if vulnerability_data['nodes'].get(node, 0) > 0.7 else 
            self.color_at_risk if vulnerability_data['nodes'].get(node, 0) > 0.3 else 
            self.color_operational for node in G.nodes()
        ]
        node_sizes = [self.node_size * (G.nodes[node].get('capacity', 100) / 100) for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color='gray', alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9)
        
        # Add labels
        labels = {node: node for node in G.nodes()}
        for node, (x, y) in pos.items():
            ax.text(x, y, node, fontsize=8, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_title('Grid Vulnerability Network')
        ax.axis('off')  # Turn off the axis
