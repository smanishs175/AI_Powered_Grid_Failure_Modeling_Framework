"""
Example script demonstrating the use of the Data Management Module.

This script shows how to initialize the module, load data, generate synthetic data,
and perform various operations on the data.
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gfmf.data_management.data_management_module import DataManagementModule

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate the Data Management Module."""
    logger.info("Starting Data Management Module example")
    
    # Initialize the module with default configuration
    dm = DataManagementModule()
    
    # Example 1: Generate and analyze synthetic data
    logger.info("Example 1: Generate and analyze synthetic data")
    
    # Generate synthetic data
    synthetic_data = dm.generate_synthetic_data(
        grid_params={'num_nodes': 30, 'num_lines': 45},
        weather_params={'num_stations': 3, 'start_date': '2023-01-01', 'end_date': '2023-06-30'},
        outage_params={'num_outages': 50},
        save=True
    )
    
    # Print summary of generated data
    print("\nSynthetic Data Summary:")
    print(f"Grid: {len(synthetic_data['grid'])} components")
    print(f"Weather: {len(synthetic_data['weather'])} records, {synthetic_data['weather']['station_id'].nunique()} stations")
    print(f"Outages: {len(synthetic_data['outage'])} outage events")
    print(f"Combined: {len(synthetic_data['combined'])} records")
    
    # Example 2: Analyze the data
    logger.info("Example 2: Analyze the data")
    
    # Analysis 1: Distribution of component types in grid
    grid_data = synthetic_data['grid']
    if 'component_type' in grid_data.columns:
        component_counts = grid_data['component_type'].value_counts()
        print("\nGrid Component Types:")
        print(component_counts)
    
    # Analysis 2: Temporal distribution of outages
    outage_data = synthetic_data['outage']
    if 'start_time' in outage_data.columns:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(outage_data['start_time']):
            outage_data['start_time'] = pd.to_datetime(outage_data['start_time'])
        
        # Count by month
        outage_data['month'] = outage_data['start_time'].dt.month
        monthly_outages = outage_data['month'].value_counts().sort_index()
        
        print("\nOutages by Month:")
        for month, count in monthly_outages.items():
            print(f"Month {month}: {count} outages")
    
    # Analysis 3: Weather-related vs. non-weather-related outages
    if 'is_weather_related' in outage_data.columns:
        weather_counts = outage_data['is_weather_related'].value_counts()
        print("\nWeather-related vs. Non-weather-related Outages:")
        print(f"Weather-related: {weather_counts.get(True, 0)}")
        print(f"Non-weather-related: {weather_counts.get(False, 0)}")
    
    # Example 3: Create some simple visualizations
    logger.info("Example 3: Create some simple visualizations")
    
    try:
        # Create a figures directory if it doesn't exist
        figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot 1: Grid topology visualization (basic)
        if 'location_x' in grid_data.columns and 'location_y' in grid_data.columns:
            plt.figure(figsize=(10, 8))
            nodes = grid_data[grid_data['component_type'] == 'node']
            lines = grid_data[grid_data['component_type'] == 'line']
            
            # Plot nodes
            plt.scatter(nodes['location_x'], nodes['location_y'], s=100, c='blue', alpha=0.7, label='Nodes')
            
            # Plot lines if we have from/to information
            if 'from_node' in lines.columns and 'to_node' in lines.columns:
                for _, line in lines.iterrows():
                    from_node = nodes[nodes['component_id'] == line['from_node']]
                    to_node = nodes[nodes['component_id'] == line['to_node']]
                    
                    if not from_node.empty and not to_node.empty:
                        plt.plot(
                            [from_node['location_x'].values[0], to_node['location_x'].values[0]],
                            [from_node['location_y'].values[0], to_node['location_y'].values[0]],
                            'k-', alpha=0.4
                        )
            
            plt.title('Synthetic Grid Topology')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figures_dir, 'grid_topology.png'))
            logger.info(f"Saved grid topology visualization to {os.path.join(figures_dir, 'grid_topology.png')}")
        
        # Plot 2: Temperature variation over time
        weather_data = synthetic_data['weather']
        if 'timestamp' in weather_data.columns and 'temperature' in weather_data.columns:
            plt.figure(figsize=(12, 6))
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(weather_data['timestamp']):
                weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            
            # Group by date if we have multiple stations
            if weather_data['station_id'].nunique() > 1:
                daily_avg = weather_data.groupby(weather_data['timestamp'].dt.date)['temperature'].mean()
                daily_avg.plot(title='Average Daily Temperature')
            else:
                weather_data.plot(x='timestamp', y='temperature', title='Temperature Over Time')
            
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figures_dir, 'temperature_variation.png'))
            logger.info(f"Saved temperature visualization to {os.path.join(figures_dir, 'temperature_variation.png')}")
        
        # Plot 3: Outage duration distribution
        if 'duration' in outage_data.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(outage_data['duration'], bins=20, alpha=0.7, color='orange')
            plt.title('Outage Duration Distribution')
            plt.xlabel('Duration (hours)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figures_dir, 'outage_duration.png'))
            logger.info(f"Saved outage duration visualization to {os.path.join(figures_dir, 'outage_duration.png')}")
        
        plt.close('all')
        
    except Exception as e:
        logger.warning(f"Error creating visualizations: {e}")
    
    # Example 4: Export data
    logger.info("Example 4: Export processed data")
    
    export_dir = os.path.join(os.path.dirname(__file__), 'exported_data')
    os.makedirs(export_dir, exist_ok=True)
    
    # Export combined data as CSV
    export_path = os.path.join(export_dir, 'combined_data.csv')
    if 'combined' in synthetic_data:
        synthetic_data['combined'].to_csv(export_path, index=False)
        logger.info(f"Exported combined data to {export_path}")
    
    logger.info("Data Management Module example completed")

if __name__ == "__main__":
    main()
