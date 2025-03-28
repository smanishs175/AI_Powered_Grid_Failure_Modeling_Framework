"""
Synthetic data generation for the Grid Failure Modeling Framework.

This module provides the class for generating synthetic data for grid topology,
weather conditions, and outage events.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Any, Tuple
import datetime
import networkx as nx
from scipy import stats
import json

logger = logging.getLogger(__name__)

class SyntheticGenerator:
    """Generator for synthetic data in the Grid Failure Modeling Framework."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SyntheticGenerator.
        
        Args:
            config: Configuration dictionary with synthetic data generation settings
        """
        self.config = config
        self.base_path = config.get('data_paths', {}).get('base_path', '')
        self.synthetic_path = config.get('data_paths', {}).get('synthetic_path', 'data/synthetic')
        self.seed = config.get('synthetic_generation', {}).get('seed', 42)
        
        # Set the random seed
        np.random.seed(self.seed)
    
    def generate_all(self, sample_data: Optional[Dict[str, Any]] = None, 
                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate all types of synthetic data.
        
        Args:
            sample_data: Optional dictionary containing sample data to base generation on
            save_path: Optional path to save synthetic data
                
        Returns:
            Dict: Dictionary containing synthetic data
        """
        synthetic_data = {}
        
        # Generate grid topology data
        logger.info("Generating synthetic grid topology data")
        synthetic_data['grid'] = self.generate_grid_topology(
            sample_grid=sample_data.get('grid', None) if sample_data else None
        )
        
        # Generate weather data
        logger.info("Generating synthetic weather data")
        synthetic_data['weather'] = self.generate_weather_data(
            sample_weather=sample_data.get('weather', None) if sample_data else None,
            grid_data=synthetic_data['grid']
        )
        
        # Generate outage data
        logger.info("Generating synthetic outage data")
        synthetic_data['outage'] = self.generate_outage_data(
            sample_outage=sample_data.get('outage', None) if sample_data else None,
            grid_data=synthetic_data['grid'],
            weather_data=synthetic_data['weather']
        )
        
        # Generate combined dataset
        logger.info("Generating synthetic combined dataset")
        synthetic_data['combined'] = self.generate_combined_data(
            grid_data=synthetic_data['grid'],
            weather_data=synthetic_data['weather'],
            outage_data=synthetic_data['outage']
        )
        
        # Save synthetic data if save_path is provided
        if save_path:
            self._save_synthetic_data(synthetic_data, save_path)
        
        return synthetic_data
    
    def generate_grid_topology(self, 
                              sample_grid: Optional[pd.DataFrame] = None,
                              num_nodes: Optional[int] = None,
                              num_lines: Optional[int] = None,
                              topology_type: str = 'mesh') -> pd.DataFrame:
        """
        Generate synthetic grid topology data.
        
        Args:
            sample_grid: Optional sample grid data to base generation on
            num_nodes: Number of nodes to generate, default is from config or 100
            num_lines: Number of lines to generate, default is derived from nodes
            topology_type: Type of topology to generate ('mesh', 'ring', 'radial')
                
        Returns:
            DataFrame: DataFrame containing synthetic grid topology
        """
        # Determine parameters
        if num_nodes is None:
            num_nodes = self.config.get('synthetic_generation', {}).get('num_nodes', 100)
        
        if num_lines is None:
            if topology_type == 'mesh':
                # For mesh, create a graph with average degree of 3
                num_lines = int(num_nodes * 1.5)
            elif topology_type == 'ring':
                # For ring, each node connects to 2 others
                num_lines = num_nodes
            elif topology_type == 'radial':
                # For radial, each node except root has 1 parent
                num_lines = num_nodes - 1
            else:
                # Default
                num_lines = int(num_nodes * 1.2)
        
        logger.info(f"Generating synthetic grid with {num_nodes} nodes and {num_lines} lines")
        
        # Create the graph
        G = None
        
        if topology_type == 'mesh':
            # Generate a random graph with specified number of nodes and edges
            G = nx.gnm_random_graph(num_nodes, num_lines, seed=self.seed)
            
            # Ensure the graph is connected
            if not nx.is_connected(G):
                # Find largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
                
                # Add edges to connect other components
                components = list(nx.connected_components(G))
                for i in range(1, len(components)):
                    # Connect first node in this component to random node in largest component
                    src = list(components[i])[0]
                    tgt = np.random.choice(list(largest_cc))
                    G.add_edge(src, tgt)
        
        elif topology_type == 'ring':
            # Create a ring topology
            G = nx.cycle_graph(num_nodes)
        
        elif topology_type == 'radial':
            # Create a tree topology
            G = nx.random_tree(num_nodes, seed=self.seed)
        
        else:
            # Default to mesh
            G = nx.gnm_random_graph(num_nodes, num_lines, seed=self.seed)
        
        # Convert graph to dataframes
        nodes_data = []
        for node in G.nodes():
            # Generate random node attributes
            node_type = np.random.choice(['bus', 'load', 'generator'], p=[0.7, 0.2, 0.1])
            voltage = np.random.choice([0.48, 4.16, 13.2, 69.0, 138.0, 230.0], p=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05])
            
            # Random location (simplified US lat/long)
            latitude = np.random.uniform(25, 49)  # US latitude range
            longitude = np.random.uniform(-125, -65)  # US longitude range
            
            # Random age between 0 and 50 years
            age = np.random.uniform(0, 50)
            
            # Random capacity
            if node_type == 'generator':
                capacity = np.random.uniform(10, 500)  # MW
            elif node_type == 'load':
                capacity = np.random.uniform(1, 50)  # MW
            else:
                capacity = np.nan
            
            nodes_data.append({
                'component_id': f'node_{node}',
                'component_type': 'node',
                'specific_type': node_type,
                'voltage': voltage,
                'age': age,
                'capacity': capacity,
                'location_x': longitude,
                'location_y': latitude,
                'degree': G.degree(node)
            })
        
        lines_data = []
        for idx, (u, v) in enumerate(G.edges()):
            # Generate random line attributes
            line_type = np.random.choice(['overhead', 'underground'], p=[0.8, 0.2])
            length = np.random.uniform(0.5, 20)  # km
            
            # Random age between 0 and 50 years
            age = np.random.uniform(0, 50)
            
            # Random capacity
            capacity = np.random.uniform(50, 500)  # MW
            
            lines_data.append({
                'component_id': f'line_{idx}',
                'component_type': 'line',
                'specific_type': line_type,
                'from': f'node_{u}',
                'to': f'node_{v}',
                'length_km': length,
                'age': age,
                'capacity': capacity
            })
        
        # Create dataframes
        nodes_df = pd.DataFrame(nodes_data)
        lines_df = pd.DataFrame(lines_data)
        
        # Combine into a single components dataframe
        components_df = pd.concat([nodes_df, lines_df], ignore_index=True)
        
        # Calculate network centrality measures
        node_centrality = nx.betweenness_centrality(G)
        edge_centrality = nx.edge_betweenness_centrality(G)
        
        # Add centrality to dataframes
        for i, row in nodes_df.iterrows():
            node_id = int(row['component_id'].split('_')[1])
            components_df.loc[components_df['component_id'] == row['component_id'], 'centrality'] = node_centrality.get(node_id, 0)
        
        for i, row in lines_df.iterrows():
            from_id = int(row['from'].split('_')[1])
            to_id = int(row['to'].split('_')[1])
            components_df.loc[components_df['component_id'] == row['component_id'], 'centrality'] = edge_centrality.get((from_id, to_id), edge_centrality.get((to_id, from_id), 0))
        
        # Calculate vulnerabilities based on age and centrality
        components_df['age_factor'] = components_df['age'] / components_df['age'].max()
        components_df['vulnerability'] = 0.3 * components_df['age_factor'] + 0.7 * components_df['centrality']
        
        logger.info(f"Generated synthetic grid with {len(nodes_df)} nodes and {len(lines_df)} lines")
        
        return components_df
    
    def generate_weather_data(self,
                             sample_weather: Optional[pd.DataFrame] = None,
                             grid_data: Optional[pd.DataFrame] = None,
                             num_stations: Optional[int] = None,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             frequency: str = 'D') -> pd.DataFrame:
        """
        Generate synthetic weather data.
        
        Args:
            sample_weather: Optional sample weather data to base generation on
            grid_data: Optional grid data to align weather stations with
            num_stations: Number of weather stations to generate, default is from config
            start_date: Start date for weather data, default is from config or 1 year ago
            end_date: End date for weather data, default is from config or current date
            frequency: Frequency of weather data ('D' for daily, 'H' for hourly)
                
        Returns:
            DataFrame: DataFrame containing synthetic weather data
        """
        # Determine parameters
        if num_stations is None:
            num_stations = self.config.get('synthetic_generation', {}).get('weather', {}).get('num_stations', 10)
        
        if start_date is None:
            start_date = self.config.get('synthetic_generation', {}).get('weather', {}).get('start_date', 
                                                                                          (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'))
        
        if end_date is None:
            end_date = self.config.get('synthetic_generation', {}).get('weather', {}).get('end_date', 
                                                                                        datetime.datetime.now().strftime('%Y-%m-%d'))
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        logger.info(f"Generating synthetic weather data from {start_date} to {end_date} for {num_stations} stations")
        
        # Create date range based on frequency
        if frequency == 'H':
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='H')
        else:
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # Generate station locations
        # If grid data is provided, distribute stations uniformly in the grid area
        if grid_data is not None and 'location_x' in grid_data.columns and 'location_y' in grid_data.columns:
            # Get grid boundaries
            min_x = grid_data['location_x'].min()
            max_x = grid_data['location_x'].max()
            min_y = grid_data['location_y'].min()
            max_y = grid_data['location_y'].max()
            
            # Add padding (10%)
            padding_x = 0.1 * (max_x - min_x)
            padding_y = 0.1 * (max_y - min_y)
            
            min_x -= padding_x
            max_x += padding_x
            min_y -= padding_y
            max_y += padding_y
            
            # Generate random station locations
            station_locs = []
            for i in range(num_stations):
                station_locs.append({
                    'station_id': f'S{i:03d}',
                    'station_name': f'Station {i}',
                    'longitude': np.random.uniform(min_x, max_x),
                    'latitude': np.random.uniform(min_y, max_y),
                    'elevation': np.random.uniform(0, 1000)  # Elevation in meters
                })
        else:
            # Generate random stations across the US
            station_locs = []
            for i in range(num_stations):
                station_locs.append({
                    'station_id': f'S{i:03d}',
                    'station_name': f'Station {i}',
                    'longitude': np.random.uniform(-125, -65),  # US longitude range
                    'latitude': np.random.uniform(25, 49),      # US latitude range
                    'elevation': np.random.uniform(0, 3000)     # Elevation in meters
                })
        
        # Create stations DataFrame
        stations_df = pd.DataFrame(station_locs)
        
        # Generate weather data for each station
        weather_data = []
        
        for _, station in stations_df.iterrows():
            station_id = station['station_id']
            lat = station['latitude']
            
            # Base temperature on latitude and time of year
            # Lower latitudes are warmer
            base_temp = 25 - 0.5 * (lat - 25)  # Base temperature decreases with latitude
            
            for date in date_range:
                # Day of year - for seasonal effects (0 to 365)
                day_of_year = date.dayofyear
                
                # Seasonal temperature variation - sine wave pattern
                # In Northern Hemisphere, warmest in summer (~day 200), coldest in winter
                seasonal_effect = 15 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
                
                # Add hourly variation if hourly data
                hourly_effect = 0
                if frequency == 'H':
                    # Hour of day (0 to 23)
                    hour = date.hour
                    # Daily temperature cycle - warmest in afternoon (~hour 14), coolest pre-dawn
                    hourly_effect = 5 * np.sin(2 * np.pi * (hour - 3) / 24)
                
                # Random daily variation
                daily_variation = np.random.normal(0, 3)
                
                # Calculate temperature
                temperature = base_temp + seasonal_effect + hourly_effect + daily_variation
                
                # Calculate min/max temperature
                if frequency == 'D':
                    # For daily data, simulate min/max around mean with random spread
                    temp_spread = np.random.uniform(3, 8)  # Random temperature spread
                    temp_min = temperature - temp_spread
                    temp_max = temperature + temp_spread
                else:
                    # For hourly data, use the same value
                    temp_min = temperature
                    temp_max = temperature
                
                # Precipitation depends on temperature and season
                # More precipitation in spring and fall
                spring_fall_factor = np.sin(4 * np.pi * day_of_year / 365) ** 2
                
                # Base precipitation probability varies by season
                precip_prob = 0.2 + 0.3 * spring_fall_factor
                
                # Precipitation amount (mm) - exponential distribution when it rains
                precipitation = 0
                if np.random.random() < precip_prob:
                    precipitation = np.random.exponential(scale=5)
                
                # Wind speed - gamma distribution
                wind_speed = np.random.gamma(shape=2, scale=2)
                
                # Humidity - beta distribution scaled to 0-100
                humidity = 100 * np.random.beta(2, 2)
                
                # Pressure - normal distribution around 1013 hPa
                pressure = np.random.normal(1013, 5)
                
                # Store the data point
                weather_data.append({
                    'station_id': station_id,
                    'timestamp': date,
                    'temperature': temperature,
                    'temperature_min': temp_min,
                    'temperature_max': temp_max,
                    'precipitation': precipitation,
                    'wind_speed': wind_speed,
                    'humidity': humidity,
                    'pressure': pressure,
                    'latitude': lat,
                    'longitude': station['longitude'],
                    'elevation': station['elevation'],
                    'station_name': station['station_name']
                })
        
        # Create DataFrame
        weather_df = pd.DataFrame(weather_data)
        
        # Add derived features
        
        # Add month and hour
        weather_df['month'] = weather_df['timestamp'].dt.month
        if frequency == 'H':
            weather_df['hour'] = weather_df['timestamp'].dt.hour
        else:
            weather_df['hour'] = 12  # Noon for daily data
        
        # Add season
        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }
        weather_df['season'] = weather_df['month'].map(season_map)
        
        # Calculate 24-hour temperature change
        weather_df = weather_df.sort_values(['station_id', 'timestamp'])
        weather_df['temperature_24h_delta'] = weather_df.groupby('station_id')['temperature'].diff(periods=24 if frequency == 'H' else 1)
        
        # Flag extreme conditions
        weather_df['is_extreme_temperature'] = (
            (weather_df['temperature'] > 35) |  # Hot (>35°C)
            (weather_df['temperature'] < -10)    # Cold (<-10°C)
        )
        
        weather_df['is_extreme_wind'] = weather_df['wind_speed'] > 30  # Strong wind (>30 mph)
        weather_df['is_extreme_precipitation'] = weather_df['precipitation'] > 50  # Heavy rain (>50mm)
        
        # Generate multi-day patterns (heat waves, cold snaps, storms)
        weather_df['heat_wave_day'] = 0
        weather_df['cold_snap_day'] = 0
        weather_df['storm_day'] = 0
        
        # Process each station separately
        for station_id, station_data in weather_df.groupby('station_id'):
            station_data = station_data.sort_values('timestamp')
            
            # Track consecutive days with extreme conditions
            for idx, row in station_data.iterrows():
                if row['is_extreme_temperature'] and row['temperature'] > 35:
                    # Potential heat wave
                    prev_day = station_data[station_data['timestamp'] < row['timestamp']].iloc[-1] if len(station_data[station_data['timestamp'] < row['timestamp']]) > 0 else None
                    
                    if prev_day is not None and prev_day['heat_wave_day'] > 0:
                        weather_df.loc[idx, 'heat_wave_day'] = prev_day['heat_wave_day'] + 1
                    else:
                        weather_df.loc[idx, 'heat_wave_day'] = 1
                
                if row['is_extreme_temperature'] and row['temperature'] < -10:
                    # Potential cold snap
                    prev_day = station_data[station_data['timestamp'] < row['timestamp']].iloc[-1] if len(station_data[station_data['timestamp'] < row['timestamp']]) > 0 else None
                    
                    if prev_day is not None and prev_day['cold_snap_day'] > 0:
                        weather_df.loc[idx, 'cold_snap_day'] = prev_day['cold_snap_day'] + 1
                    else:
                        weather_df.loc[idx, 'cold_snap_day'] = 1
                
                if row['is_extreme_precipitation'] or row['is_extreme_wind']:
                    # Potential storm
                    prev_day = station_data[station_data['timestamp'] < row['timestamp']].iloc[-1] if len(station_data[station_data['timestamp'] < row['timestamp']]) > 0 else None
                    
                    if prev_day is not None and prev_day['storm_day'] > 0:
                        weather_df.loc[idx, 'storm_day'] = prev_day['storm_day'] + 1
                    else:
                        weather_df.loc[idx, 'storm_day'] = 1
        
        logger.info(f"Generated synthetic weather data with {len(weather_df)} records for {num_stations} stations")
        
        return weather_df
    
    def generate_outage_data(self,
                        sample_outage: Optional[pd.DataFrame] = None,
                        grid_data: Optional[pd.DataFrame] = None,
                        weather_data: Optional[pd.DataFrame] = None,
                        num_outages: Optional[int] = None,
                        baseline_rate: float = 0.01) -> pd.DataFrame:
        """
        Generate synthetic outage data.
        
        Args:
            sample_outage: Optional sample outage data to base generation on
            grid_data: Grid data to associate outages with components
            weather_data: Weather data to correlate outages with conditions
            num_outages: Number of outages to generate, default is calculated from baseline rate
            baseline_rate: Baseline annual outage rate per component
                
        Returns:
            DataFrame: DataFrame containing synthetic outage data
        """
        if grid_data is None or weather_data is None:
            logger.error("Grid data and weather data are required to generate outages")
            return pd.DataFrame()
        
        logger.info("Generating synthetic outage data")
        
        # Calculate time range from weather data
        weather_data = weather_data.copy()
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
        start_date = weather_data['timestamp'].min()
        end_date = weather_data['timestamp'].max()
        date_range_days = (end_date - start_date).days
        
        # Get unique components from grid data
        components = grid_data[['component_id', 'component_type', 'specific_type', 'age', 'centrality']]
        
        # If no specified number of outages, calculate based on baseline rate
        if num_outages is None:
            # Expected outages = components × baseline rate × (days / 365)
            expected_outages = len(components) * baseline_rate * (date_range_days / 365)
            num_outages = int(expected_outages)
            
            # Minimum of 5 outages for small datasets
            num_outages = max(num_outages, 5)
        
        logger.info(f"Generating {num_outages} synthetic outages over {date_range_days} days")
        
        # Create outage probabilities for each component based on attributes
        # Older components and those with higher centrality are more likely to fail
        if 'age' in components.columns and 'centrality' in components.columns:
            # Normalize age and centrality to 0-1 range
            max_age = components['age'].max()
            if max_age > 0:
                components['age_factor'] = components['age'] / max_age
            else:
                components['age_factor'] = 0
                
            max_centrality = components['centrality'].max()
            if max_centrality > 0:
                components['centrality_factor'] = components['centrality'] / max_centrality
            else:
                components['centrality_factor'] = 0
                
            # Calculate outage probability
            components['outage_prob'] = 0.4 * components['age_factor'] + 0.6 * components['centrality_factor']
            
            # Normalize to ensure sum = 1
            components['outage_prob'] = components['outage_prob'] / components['outage_prob'].sum()
        else:
            # Equal probability for all components
            components['outage_prob'] = 1.0 / len(components)
        
        # Generate outages
        outage_data = []
        outage_id_counter = 0
        
        # Group weather data by day to make lookup easier
        weather_by_day = {}
        for day, day_data in weather_data.groupby(weather_data['timestamp'].dt.date):
            weather_by_day[day] = day_data
        
        # Generate non-weather related outages
        num_regular_outages = int(num_outages * 0.7)  # 70% are regular outages
        
        for _ in range(num_regular_outages):
            # Select component based on failure probability
            component = components.sample(n=1, weights='outage_prob').iloc[0]
            
            # Random date within range
            days_offset = np.random.randint(0, date_range_days)
            outage_date = start_date + datetime.timedelta(days=days_offset)
            
            # Random time of day (more likely during peak hours)
            hour_weights = np.ones(24)
            # Increase probability during 7-9 AM and 5-7 PM
            hour_weights[7:10] = 2
            hour_weights[17:20] = 2
            hour = np.random.choice(range(24), p=hour_weights/hour_weights.sum())
            
            outage_datetime = datetime.datetime.combine(outage_date.date(), datetime.time(hour=hour))
            
            # Duration based on component type (lines typically take longer to repair than nodes)
            if component['component_type'] == 'line':
                duration = np.random.exponential(scale=8)  # Mean of 8 hours
            else:
                duration = np.random.exponential(scale=4)  # Mean of 4 hours
            
            # Ensure reasonable duration (at least 30 minutes, at most 72 hours)
            duration = max(0.5, min(72, duration))
            
            # Impact (affected customers) depends on component centrality
            if 'centrality' in component:
                impact_base = 1000 * (0.1 + 0.9 * component['centrality_factor'])
                impact = int(impact_base * np.random.uniform(0.5, 1.5))  # Add some randomness
            else:
                impact = np.random.randint(100, 5000)
            
            # Create the outage record
            outage_data.append({
                'outage_id': f"O{outage_id_counter:06d}",
                'component_id': component['component_id'],
                'component_type': component['component_type'],
                'start_time': outage_datetime,
                'duration': duration,
                'end_time': outage_datetime + datetime.timedelta(hours=duration),
                'cause': 'equipment_failure',
                'is_weather_related': False,
                'impact': impact,
                'cascading': False,
                'preceded_by': [],
                'followed_by': []
            })
            
            outage_id_counter += 1
        
        # Generate weather-related outages
        num_weather_outages = num_outages - num_regular_outages
        
        # Find days with extreme weather
        extreme_days = []
        for day, day_data in weather_by_day.items():
            if (day_data['is_extreme_temperature'].any() or 
                day_data['is_extreme_wind'].any() or 
                day_data['is_extreme_precipitation'].any()):
                extreme_days.append(day)
        
        if not extreme_days:
            # If no extreme days, just pick random days
            available_days = list(weather_by_day.keys())
            extreme_days = np.random.choice(available_days, 
                                          size=min(10, len(available_days)), 
                                          replace=False)
        
        # Generate weather-related outages on extreme days
        for _ in range(num_weather_outages):
            # Select a day with extreme weather
            outage_day = np.random.choice(extreme_days)
            day_weather = weather_by_day[outage_day]
            
            # Select component based on failure probability
            component = components.sample(n=1, weights='outage_prob').iloc[0]
            
            # If there was extreme wind on this day, lines are more likely to fail
            if 'is_extreme_wind' in day_weather.columns and day_weather['is_extreme_wind'].any():
                if np.random.random() < 0.7:  # 70% chance to pick a line during wind events
                    line_components = components[components['component_type'] == 'line']
                    if not line_components.empty:
                        component = line_components.sample(n=1, weights='outage_prob').iloc[0]
            
            # Select time during day (more likely during most extreme conditions)
            if 'is_extreme_temperature' in day_weather.columns and day_weather['is_extreme_temperature'].any():
                # During extreme temperatures, outages more likely during temperature extremes
                if day_weather['temperature'].max() > 35:  # Heat
                    # More likely in afternoon (peak heat)
                    hour_weights = np.ones(24)
                    hour_weights[12:18] = 3  # Afternoon hours
                    hour = np.random.choice(range(24), p=hour_weights/hour_weights.sum())
                elif day_weather['temperature'].min() < -10:  # Cold
                    # More likely in evening/night (peak cold)
                    hour_weights = np.ones(24)
                    hour_weights[18:24] = 3  # Evening hours
                    hour_weights[0:6] = 3    # Night hours
                    hour = np.random.choice(range(24), p=hour_weights/hour_weights.sum())
                else:
                    hour = np.random.randint(0, 24)
            else:
                hour = np.random.randint(0, 24)
            
            outage_datetime = datetime.datetime.combine(outage_day, datetime.time(hour=hour))
            
            # Duration tends to be longer during extreme weather
            if component['component_type'] == 'line':
                duration = np.random.exponential(scale=12)  # Mean of 12 hours
            else:
                duration = np.random.exponential(scale=6)  # Mean of 6 hours
            
            # Ensure reasonable duration (at least 1 hour, at most 96 hours for weather-related)
            duration = max(1, min(96, duration))
            
            # Impact tends to be higher during extreme weather
            if 'centrality' in component:
                impact_base = 2000 * (0.2 + 0.8 * component['centrality_factor'])
                impact = int(impact_base * np.random.uniform(0.8, 2.0))  # More randomness, higher impact
            else:
                impact = np.random.randint(500, 10000)
            
            # Determine cause based on weather
            if 'is_extreme_wind' in day_weather.columns and day_weather['is_extreme_wind'].any():
                cause = 'wind'
            elif 'is_extreme_precipitation' in day_weather.columns and day_weather['is_extreme_precipitation'].any():
                cause = 'precipitation'
            elif 'is_extreme_temperature' in day_weather.columns and day_weather['is_extreme_temperature'].any():
                if day_weather['temperature'].max() > 35:
                    cause = 'heat'
                else:
                    cause = 'cold'
            else:
                cause = 'weather_other'
            
            # Create the outage record
            outage_data.append({
                'outage_id': f"O{outage_id_counter:06d}",
                'component_id': component['component_id'],
                'component_type': component['component_type'],
                'start_time': outage_datetime,
                'duration': duration,
                'end_time': outage_datetime + datetime.timedelta(hours=duration),
                'cause': cause,
                'is_weather_related': True,
                'impact': impact,
                'cascading': False,
                'preceded_by': [],
                'followed_by': []
            })
            
            outage_id_counter += 1
        
        # Create DataFrame
        outage_df = pd.DataFrame(outage_data)
        
        # Sort by start time
        outage_df = outage_df.sort_values('start_time')
        
        # Add cascading outage effects
        # Identify outages that start within 1 hour of another outage in nearby components
        cascading_window = datetime.timedelta(hours=1)
        
        for i, outage in outage_df.iterrows():
            # Find potential cascading outages
            cascade_candidates = outage_df[
                (outage_df['start_time'] > outage['start_time']) & 
                (outage_df['start_time'] <= outage['start_time'] + cascading_window) &
                (outage_df['outage_id'] != outage['outage_id'])
            ]
            
            if not cascade_candidates.empty:
                # Some outages may cascade from this one
                outage_df.at[i, 'followed_by'] = cascade_candidates['outage_id'].tolist()
                
                # Mark those outages as cascading and preceded by this one
                for j, cascade in cascade_candidates.iterrows():
                    # 50% chance that each candidate is actually a cascading outage
                    if np.random.random() < 0.5:
                        outage_df.at[j, 'cascading'] = True
                        outage_df.at[j, 'preceded_by'] = [outage['outage_id']]
        
        logger.info(f"Generated {len(outage_df)} synthetic outages ({outage_df['is_weather_related'].sum()} weather-related)")
        
        return outage_df
    
    def generate_combined_data(self,
                          grid_data: pd.DataFrame,
                          weather_data: pd.DataFrame,
                          outage_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a combined dataset aligning grid, weather, and outage data.
        
        Args:
            grid_data: DataFrame containing grid component data
            weather_data: DataFrame containing weather data
            outage_data: DataFrame containing outage data
                
        Returns:
            DataFrame: Combined dataset with aligned timestamps
        """
        logger.info("Generating combined dataset")
        
        # Create a cross-join of components and timestamps
        # Get unique timestamps from weather data
        unique_timestamps = weather_data['timestamp'].unique()
        
        # Get components
        components = grid_data[['component_id', 'component_type', 'specific_type', 'age', 'centrality', 'location_x', 'location_y']]
        
        # Create empty combined dataset
        combined_data = []
        
        # Sample timestamps (using a subset to reduce dataset size)
        # For daily data, use every day; for hourly data, sample every 6 hours
        is_hourly = len(unique_timestamps) > 1000  # Approximate check
        
        if is_hourly:
            # Sample every 6 hours to reduce size
            timestamps_to_use = sorted([ts for ts in unique_timestamps if pd.to_datetime(ts).hour % 6 == 0])
        else:
            timestamps_to_use = sorted(unique_timestamps)
        
        # Limit to max 1000 timestamps for performance
        if len(timestamps_to_use) > 1000:
            timestamps_to_use = np.random.choice(timestamps_to_use, size=1000, replace=False)
            timestamps_to_use = sorted(timestamps_to_use)
        
        logger.info(f"Using {len(timestamps_to_use)} timestamps for combined dataset")
        
        # Create nearest neighbor lookup for weather stations
        # Get unique stations with their coordinates
        stations = weather_data[['station_id', 'latitude', 'longitude']].drop_duplicates()
        station_coords = stations[['latitude', 'longitude']].values
        
        for component_idx, component in components.iterrows():
            # Get component coordinates
            if pd.notna(component['location_x']) and pd.notna(component['location_y']):
                comp_lat = component['location_y']
                comp_lon = component['location_x']
                
                # Find nearest weather station
                if not stations.empty:
                    # Calculate distances to all stations
                    distances = np.sqrt(
                        (station_coords[:, 0] - comp_lat) ** 2 + 
                        (station_coords[:, 1] - comp_lon) ** 2
                    )
                    nearest_idx = np.argmin(distances)
                    nearest_station = stations.iloc[nearest_idx]['station_id']
                else:
                    nearest_station = None
            else:
                nearest_station = None
            
            # Process each timestamp
            for ts in timestamps_to_use:
                # Get weather data for this timestamp and station
                if nearest_station is not None:
                    weather_at_ts = weather_data[
                        (weather_data['timestamp'] == ts) & 
                        (weather_data['station_id'] == nearest_station)
                    ]
                else:
                    # If no station, just get any weather data for this timestamp
                    weather_at_ts = weather_data[weather_data['timestamp'] == ts]
                
                # Default weather values
                temp = np.nan
                wind = np.nan
                precip = np.nan
                is_extreme = False
                
                # Extract weather variables if data exists
                if not weather_at_ts.empty:
                    temp = weather_at_ts['temperature'].values[0]
                    
                    if 'wind_speed' in weather_at_ts.columns:
                        wind = weather_at_ts['wind_speed'].values[0]
                    
                    if 'precipitation' in weather_at_ts.columns:
                        precip = weather_at_ts['precipitation'].values[0]
                    
                    # Check if any extreme conditions
                    if ('is_extreme_temperature' in weather_at_ts.columns and 
                        'is_extreme_wind' in weather_at_ts.columns and 
                        'is_extreme_precipitation' in weather_at_ts.columns):
                        is_extreme = (
                            weather_at_ts['is_extreme_temperature'].values[0] or
                            weather_at_ts['is_extreme_wind'].values[0] or
                            weather_at_ts['is_extreme_precipitation'].values[0]
                        )
                
                # Check for outages in the next 24 hours for this component
                ts_dt = pd.to_datetime(ts)
                next_24h = ts_dt + datetime.timedelta(hours=24)
                
                outages_next_24h = outage_data[
                    (outage_data['component_id'] == component['component_id']) &
                    (outage_data['start_time'] >= ts_dt) &
                    (outage_data['start_time'] < next_24h)
                ]
                
                outage_occurred = not outages_next_24h.empty
                
                # Time to failure (hours)
                if outage_occurred:
                    # Time from current timestamp to next outage start
                    next_outage_time = outages_next_24h['start_time'].min()
                    time_to_failure = (next_outage_time - ts_dt).total_seconds() / 3600
                else:
                    time_to_failure = np.nan
                
                # Create the combined record
                combined_data.append({
                    'timestamp': ts,
                    'component_id': component['component_id'],
                    'component_type': component['component_type'],
                    'specific_type': component['specific_type'],
                    'age': component['age'],
                    'centrality': component['centrality'],
                    'location_x': component['location_x'],
                    'location_y': component['location_y'],
                    'temperature': temp,
                    'wind_speed': wind,
                    'precipitation': precip,
                    'is_extreme_weather': is_extreme,
                    'outage_occurred': outage_occurred,
                    'time_to_failure': time_to_failure
                })
        
        # Create DataFrame
        combined_df = pd.DataFrame(combined_data)
        
        # Add timestamp-derived columns
        combined_df['month'] = pd.to_datetime(combined_df['timestamp']).dt.month
        combined_df['hour'] = pd.to_datetime(combined_df['timestamp']).dt.hour
        combined_df['day_of_week'] = pd.to_datetime(combined_df['timestamp']).dt.dayofweek
        combined_df['is_weekend'] = combined_df['day_of_week'] >= 5
        
        logger.info(f"Generated combined dataset with {len(combined_df)} records")
        
        return combined_df
    
    def _save_synthetic_data(self, synthetic_data: Dict[str, pd.DataFrame], save_path: str) -> None:
        """
        Save synthetic data to files.
        
        Args:
            synthetic_data: Dictionary containing synthetic data DataFrames
            save_path: Path to save the data to
        """
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save grid data
        if 'grid' in synthetic_data:
            # Save as JSON
            grid_json_path = os.path.join(save_path, 'synthetic_grid.json')
            grid_data = synthetic_data['grid']
            
            # Split into nodes and lines
            nodes = grid_data[grid_data['component_type'] == 'node']
            lines = grid_data[grid_data['component_type'] == 'line']
            
            # Convert to dictionary format
            grid_dict = {
                'nodes': {},
                'lines': {}
            }
            
            for _, node in nodes.iterrows():
                node_id = node['component_id']
                grid_dict['nodes'][node_id] = node.to_dict()
            
            for _, line in lines.iterrows():
                line_id = line['component_id']
                grid_dict['lines'][line_id] = line.to_dict()
            
            # Save as JSON
            with open(grid_json_path, 'w') as f:
                json.dump(grid_dict, f, indent=2, default=str)
            
            logger.info(f"Saved synthetic grid data to {grid_json_path}")
            
            # Also save as CSV
            grid_csv_path = os.path.join(save_path, 'synthetic_grid.csv')
            grid_data.to_csv(grid_csv_path, index=False)
            logger.info(f"Saved synthetic grid data to {grid_csv_path}")
        
        # Save weather data
        if 'weather' in synthetic_data:
            weather_csv_path = os.path.join(save_path, 'synthetic_weather.csv')
            synthetic_data['weather'].to_csv(weather_csv_path, index=False)
            logger.info(f"Saved synthetic weather data to {weather_csv_path}")
        
        # Save outage data
        if 'outage' in synthetic_data:
            outage_csv_path = os.path.join(save_path, 'synthetic_outages.csv')
            # Convert list columns to strings for CSV
            outage_data = synthetic_data['outage'].copy()
            if 'preceded_by' in outage_data.columns:
                outage_data['preceded_by'] = outage_data['preceded_by'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
            if 'followed_by' in outage_data.columns:
                outage_data['followed_by'] = outage_data['followed_by'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
            outage_data.to_csv(outage_csv_path, index=False)
            logger.info(f"Saved synthetic outage data to {outage_csv_path}")
        
        # Save combined dataset
        if 'combined' in synthetic_data:
            combined_csv_path = os.path.join(save_path, 'synthetic_combined.csv')
            synthetic_data['combined'].to_csv(combined_csv_path, index=False)
            logger.info(f"Saved synthetic combined data to {combined_csv_path}")
        
        # Create a README file with data description
        readme_path = os.path.join(save_path, 'README.md')
        with open(readme_path, 'w') as f:
            f.write("# Synthetic Data for Grid Failure Modeling\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Contents\n\n")
            
            if 'grid' in synthetic_data:
                grid_data = synthetic_data['grid']
                nodes = grid_data[grid_data['component_type'] == 'node']
                lines = grid_data[grid_data['component_type'] == 'line']
                f.write(f"- **synthetic_grid.json**: Grid topology with {len(nodes)} nodes and {len(lines)} lines\n")
                f.write(f"- **synthetic_grid.csv**: Same grid data in CSV format\n")
            
            if 'weather' in synthetic_data:
                weather_data = synthetic_data['weather']
                stations = weather_data['station_id'].nunique()
                start_date = weather_data['timestamp'].min().strftime('%Y-%m-%d')
                end_date = weather_data['timestamp'].max().strftime('%Y-%m-%d')
                f.write(f"- **synthetic_weather.csv**: Weather data for {stations} stations from {start_date} to {end_date}\n")
            
            if 'outage' in synthetic_data:
                outage_data = synthetic_data['outage']
                f.write(f"- **synthetic_outages.csv**: {len(outage_data)} outage events\n")
            
            if 'combined' in synthetic_data:
                combined_data = synthetic_data['combined']
                f.write(f"- **synthetic_combined.csv**: Combined dataset with {len(combined_data)} records\n")
        
        logger.info(f"Saved synthetic data description to {readme_path}")
