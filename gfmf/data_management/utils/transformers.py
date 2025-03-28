"""
Data transformation utilities for the Data Management Module.

This module provides functions to transform and engineer features 
from different types of data used in the Grid Failure Modeling Framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Any
import networkx as nx
from sklearn.preprocessing import StandardScaler
import datetime
from dateutil.relativedelta import relativedelta

def transform_grid_topology_data(grid_data: Dict, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Transform grid topology data into a unified component DataFrame with derived features.
    
    Args:
        grid_data: Dictionary containing grid topology data (nodes and lines)
        config: Optional configuration dictionary with preprocessing settings
        
    Returns:
        DataFrame: Combined grid components with derived features
    """
    if grid_data is None or not grid_data:
        return pd.DataFrame()
    
    if config is None:
        config = {}
    
    # Extract nodes and lines DataFrames
    nodes_df = grid_data.get('nodes', pd.DataFrame())
    lines_df = grid_data.get('lines', pd.DataFrame())
    
    # Check if we have a raw dictionary instead of DataFrames
    if isinstance(nodes_df, dict):
        nodes_df = pd.DataFrame.from_dict(nodes_df, orient='index')
    if isinstance(lines_df, dict):
        lines_df = pd.DataFrame.from_dict(lines_df, orient='index')
    
    # Create copies to avoid modifying originals
    nodes = nodes_df.copy()
    lines = lines_df.copy()
    
    # Add component_type field to identify components
    nodes['component_type'] = 'node'
    lines['component_type'] = 'line'
    
    # Rename specific type fields for consistency
    if 'type' in nodes.columns:
        nodes = nodes.rename(columns={'type': 'specific_type'})
    if 'type' in lines.columns:
        lines = lines.rename(columns={'type': 'specific_type'})
    
    # Extract network properties using NetworkX
    G = create_network_graph(nodes, lines)
    
    # Calculate node centrality
    node_centrality = nx.betweenness_centrality(G)
    nodes['centrality'] = nodes.index.map(lambda x: node_centrality.get(x, 0))
    
    # Calculate edge centrality
    edge_centrality = nx.edge_betweenness_centrality(G)
    lines['centrality'] = lines.apply(
        lambda row: edge_centrality.get((row['from'], row['to']), 
                                        edge_centrality.get((row['to'], row['from']), 0)),
        axis=1
    )
    
    # Calculate connected components for redundancy measure
    connected_components = list(nx.connected_components(G))
    component_map = {}
    for i, component in enumerate(connected_components):
        for node in component:
            component_map[node] = i
    
    nodes['component_group'] = nodes.index.map(lambda x: component_map.get(x, -1))
    
    # Calculate node degree for connectivity measure
    node_degree = dict(G.degree())
    nodes['degree'] = nodes.index.map(lambda x: node_degree.get(x, 0))
    
    # Calculate redundancy factor
    # Higher values mean more alternative paths
    nodes['redundancy'] = nodes['degree'] / nodes['degree'].max()
    lines['redundancy'] = lines.apply(
        lambda row: min(
            nodes.loc[nodes.index == row['from'], 'redundancy'].values[0],
            nodes.loc[nodes.index == row['to'], 'redundancy'].values[0]
        ) if row['from'] in nodes.index and row['to'] in nodes.index else 0,
        axis=1
    )
    
    # Create age_factor based on component age
    # Older components are more likely to fail
    if 'age' in nodes.columns:
        max_age = nodes['age'].max()
        if max_age > 0:
            nodes['age_factor'] = nodes['age'] / max_age
        else:
            nodes['age_factor'] = 0
    else:
        nodes['age'] = np.nan
        nodes['age_factor'] = np.nan
        
    if 'age' in lines.columns:
        max_age = lines['age'].max()
        if max_age > 0:
            lines['age_factor'] = lines['age'] / max_age
        else:
            lines['age_factor'] = 0
    else:
        lines['age'] = np.nan
        lines['age_factor'] = np.nan
    
    # Add connected_to list for each component
    nodes['connected_to'] = nodes.index.map(
        lambda x: list(G.neighbors(x)) if x in G else []
    )
    
    lines['connected_to'] = lines.apply(
        lambda row: [row['from'], row['to']],
        axis=1
    )
    
    # Initialize utilization_avg as unknown
    nodes['utilization_avg'] = np.nan
    lines['utilization_avg'] = np.nan
    
    # Initialize historical failure rate as unknown
    nodes['failure_rate_historical'] = np.nan
    lines['failure_rate_historical'] = np.nan
    
    # Prepare for component DataFrame
    # Rename index to component_id for clarity
    nodes = nodes.reset_index().rename(columns={'index': 'component_id'})
    lines = lines.reset_index().rename(columns={'index': 'component_id'})
    
    # Ensure location data exists
    if 'LATITUDE' in nodes.columns and 'LONGITUDE' in nodes.columns:
        nodes = nodes.rename(columns={'LATITUDE': 'location_y', 'LONGITUDE': 'location_x'})
    elif 'lat' in nodes.columns and 'lon' in nodes.columns:
        nodes = nodes.rename(columns={'lat': 'location_y', 'lon': 'location_x'})
    else:
        nodes['location_x'] = np.nan
        nodes['location_y'] = np.nan
    
    # For lines, use midpoint of connected nodes as location
    lines['location_x'] = np.nan
    lines['location_y'] = np.nan
    
    # Combine nodes and lines into a single DataFrame
    components_df = pd.concat([nodes, lines], ignore_index=True)
    
    return components_df

def create_network_graph(nodes_df: pd.DataFrame, lines_df: pd.DataFrame) -> nx.Graph:
    """
    Create a NetworkX graph from grid topology data.
    
    Args:
        nodes_df: DataFrame containing node data
        lines_df: DataFrame containing line data
        
    Returns:
        nx.Graph: NetworkX graph of the grid topology
    """
    G = nx.Graph()
    
    # Add nodes
    for node_id in nodes_df.index:
        G.add_node(node_id)
    
    # Add edges
    for _, line in lines_df.iterrows():
        if 'from' in line and 'to' in line:
            G.add_edge(line['from'], line['to'])
    
    return G

def transform_weather_data(weather_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Transform weather data and engineer features.
    
    Args:
        weather_df: DataFrame containing weather data
        config: Configuration dictionary with preprocessing settings
        
    Returns:
        DataFrame: Transformed weather data with engineered features
    """
    # Create a copy to avoid modifying the original
    weather = weather_df.copy()
    
    # Ensure timestamp column is datetime
    weather['timestamp'] = pd.to_datetime(weather['DATE'])
    
    # Add month and hour
    weather['month'] = weather['timestamp'].dt.month
    weather['hour'] = weather['timestamp'].dt.hour
    
    # Add season
    # Northern hemisphere seasons
    # Winter: Dec, Jan, Feb (12, 1, 2)
    # Spring: Mar, Apr, May (3, 4, 5)
    # Summer: Jun, Jul, Aug (6, 7, 8)
    # Fall: Sep, Oct, Nov (9, 10, 11)
    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    weather['season'] = weather['month'].map(season_map)
    
    # Handle different column name conventions
    rename_map = {
        'PRCP': 'precipitation',
        'TMAX': 'temperature_max',
        'TMIN': 'temperature_min',
        'TAVG': 'temperature',
        'AWND': 'wind_speed',
        'STATION': 'station_id',
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
        'ELEVATION': 'elevation',
        'NAME': 'station_name'
    }
    
    weather = weather.rename(columns={col: rename_map[col] for col in rename_map if col in weather.columns})
    
    # Calculate average temperature if only min/max available
    if 'temperature' not in weather.columns and all(col in weather.columns for col in ['temperature_min', 'temperature_max']):
        weather['temperature'] = (weather['temperature_min'] + weather['temperature_max']) / 2
    
    # Ensure essential weather columns exist
    essential_columns = ['precipitation', 'temperature', 'wind_speed']
    for col in essential_columns:
        if col not in weather.columns:
            weather[col] = np.nan
    
    # Handle missing values according to strategy in config
    missing_strategy = config.get('preprocessing', {}).get('missing_strategy', 'interpolate')
    if missing_strategy == 'interpolate':
        # Interpolate within each station
        weather = weather.groupby('station_id').apply(
            lambda x: x[essential_columns].interpolate(method='linear')
        ).reset_index(drop=True)
    elif missing_strategy == 'ffill':
        # Forward fill within each station
        weather = weather.groupby('station_id').apply(
            lambda x: x[essential_columns].fillna(method='ffill')
        ).reset_index(drop=True)
    elif missing_strategy == 'mean':
        # Use station means
        for col in essential_columns:
            station_means = weather.groupby('station_id')[col].transform('mean')
            weather[col] = weather[col].fillna(station_means)
    
    # Calculate 24-hour temperature change
    weather = weather.sort_values(['station_id', 'timestamp'])
    weather['temperature_24h_delta'] = weather.groupby('station_id')['temperature'].diff(periods=1)
    
    # Flag extreme conditions
    # These thresholds can be adjusted based on local climate
    weather['is_extreme_temperature'] = (
        (weather['temperature'] > 35) |  # Hot (>35°C)
        (weather['temperature'] < -10)    # Cold (<-10°C)
    )
    
    weather['is_extreme_wind'] = weather['wind_speed'] > 30  # Strong wind (>30 mph)
    weather['is_extreme_precipitation'] = weather['precipitation'] > 50  # Heavy rain (>50mm)
    
    # Identify multi-day patterns
    # Heat waves (3+ consecutive days with high temps)
    # Cold snaps (3+ consecutive days with low temps)
    # Storms (high precipitation and/or wind)
    
    # Initialize pattern columns
    weather['heat_wave_day'] = 0
    weather['cold_snap_day'] = 0
    weather['storm_day'] = 0
    
    # Process each station separately
    for station_id, station_data in weather.groupby('station_id'):
        # Convert to daily data for pattern detection (if not already daily)
        if len(station_data) > 366 * 10:  # If more than ~10 years of daily data, likely hourly
            daily_data = station_data.resample('D', on='timestamp').agg({
                'temperature': 'mean',
                'precipitation': 'sum',
                'wind_speed': 'max'
            })
        else:
            daily_data = station_data.set_index('timestamp')
        
        # Heat wave detection (3+ days above 35°C)
        heat_wave = (daily_data['temperature'] > 35).astype(int)
        for i in range(1, len(heat_wave)):
            if heat_wave.iloc[i] == 1:
                heat_wave.iloc[i] += heat_wave.iloc[i-1]
        
        # Cold snap detection (3+ days below -10°C)
        cold_snap = (daily_data['temperature'] < -10).astype(int)
        for i in range(1, len(cold_snap)):
            if cold_snap.iloc[i] == 1:
                cold_snap.iloc[i] += cold_snap.iloc[i-1]
        
        # Storm detection (heavy rain or strong wind)
        storm = ((daily_data['precipitation'] > 50) | 
                (daily_data['wind_speed'] > 30)).astype(int)
        for i in range(1, len(storm)):
            if storm.iloc[i] == 1:
                storm.iloc[i] += storm.iloc[i-1]
        
        # Map back to original data
        for idx, row in station_data.iterrows():
            date = row['timestamp'].date()
            if date in heat_wave.index:
                weather.loc[idx, 'heat_wave_day'] = heat_wave[date] if heat_wave[date] > 0 else 0
            if date in cold_snap.index:
                weather.loc[idx, 'cold_snap_day'] = cold_snap[date] if cold_snap[date] > 0 else 0
            if date in storm.index:
                weather.loc[idx, 'storm_day'] = storm[date] if storm[date] > 0 else 0
    
    # Optional standardization
    if config.get('preprocessing', {}).get('standardization', False):
        numeric_columns = weather.select_dtypes(include=['float64', 'int64']).columns
        exclude_columns = ['timestamp', 'month', 'hour', 'heat_wave_day', 'cold_snap_day', 'storm_day']
        columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
        
        scaler = StandardScaler()
        weather[columns_to_scale] = scaler.fit_transform(weather[columns_to_scale])
    
    return weather

def transform_outage_data(outage_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Transform outage data and engineer features.
    
    Args:
        outage_df: DataFrame containing outage data
        config: Configuration dictionary with preprocessing settings
        
    Returns:
        DataFrame: Transformed outage data with engineered features
    """
    # Create a copy to avoid modifying the original
    outage = outage_df.copy()
    
    # Determine the type of outage data (merged or aggregated)
    is_merged_format = 'start_time' in outage.columns
    
    if is_merged_format:
        # Process detailed outage records
        
        # Ensure datetime format
        outage['start_time'] = pd.to_datetime(outage['start_time'])
        
        # Calculate end time
        outage['end_time'] = outage['start_time'] + pd.to_timedelta(outage['duration'], unit='h')
        
        # Create unique outage_id
        if 'outage_id' not in outage.columns:
            outage['outage_id'] = [f"outage_{i}" for i in range(len(outage))]
        
        # Create component_id if not present
        if 'component_id' not in outage.columns:
            # Use county as a proxy for component
            outage['component_id'] = outage['county'] + '_' + outage['state']
        
        # Initialize derived features
        outage['is_weather_related'] = False
        outage['cascading'] = False
        outage['preceded_by'] = [[] for _ in range(len(outage))]
        outage['followed_by'] = [[] for _ in range(len(outage))]
        
        # Calculate impact based on customers affected
        if all(col in outage.columns for col in ['min_customers', 'max_customers', 'mean_customers']):
            outage['impact'] = outage['mean_customers']
        elif 'customers_affected' in outage.columns:
            outage['impact'] = outage['customers_affected']
        else:
            outage['impact'] = np.nan
            
        # Determine cascading outage events
        # Consider outages as cascading if they occurred in neighboring areas within a short time
        time_threshold = pd.Timedelta(hours=1)  # Cascading events occur within 1 hour
        
        # Sort by start time
        outage = outage.sort_values('start_time')
        
        # Group by state to find potential cascading events
        for state, state_outages in outage.groupby('state'):
            state_outages = state_outages.sort_values('start_time')
            
            # Identify cascading events
            for i, row in state_outages.iterrows():
                # Find outages that started shortly after this one
                cascading_candidates = state_outages[
                    (state_outages['start_time'] > row['start_time']) & 
                    (state_outages['start_time'] <= row['start_time'] + time_threshold)
                ]
                
                if len(cascading_candidates) > 0:
                    # This outage has followers (potential cascade)
                    outage.at[i, 'followed_by'] = cascading_candidates['outage_id'].tolist()
                    
                    # Mark followers as cascading and add this outage as predecessor
                    for cascade_idx, cascade_row in cascading_candidates.iterrows():
                        outage.at[cascade_idx, 'cascading'] = True
                        outage.at[cascade_idx, 'preceded_by'] = [row['outage_id']]
        
    else:
        # Process aggregated outage records
        
        # Create dummy outage records from aggregated data
        # Note: This creates simplified records without detailed timing
        detailed_records = []
        
        for _, row in outage.iterrows():
            state = row['state']
            year = int(row['year'])
            month = int(row['month'])
            outage_count = int(row['outage_count'])
            
            # Get maximum duration if available
            if 'max_outage_duration' in row:
                duration = float(row['max_outage_duration'])
            else:
                duration = 1.0  # Default 1 hour
            
            # Get customer impact if available
            if 'customer_weighted_hours' in row:
                total_impact = float(row['customer_weighted_hours'])
                impact_per_outage = total_impact / outage_count if outage_count > 0 else 0
            else:
                impact_per_outage = np.nan
            
            # Create outage_count records spread throughout the month
            for i in range(outage_count):
                # Spread outages throughout the month
                day = (i % 28) + 1
                hour = (i % 24)
                
                # Create start time
                start_time = datetime.datetime(year, month + 1, day, hour)
                
                # Create end time
                end_time = start_time + datetime.timedelta(hours=duration)
                
                # Create record
                record = {
                    'outage_id': f"{state}_{year}_{month}_{i}",
                    'component_id': f"{state}_{i % 10}",  # Spread across 10 components
                    'state': state,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'impact': impact_per_outage,
                    'is_weather_related': False,
                    'cascading': False,
                    'preceded_by': [],
                    'followed_by': []
                }
                
                detailed_records.append(record)
        
        # Create new DataFrame with detailed records
        outage = pd.DataFrame(detailed_records)
    
    return outage

def align_datasets(grid_data: Dict, weather_df: pd.DataFrame, outage_df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Create a combined dataset aligning grid, weather, and outage data.
    
    Args:
        grid_data: Dictionary containing grid topology data (nodes and lines)
        weather_df: DataFrame containing weather data
        outage_df: DataFrame containing outage data
        config: Optional configuration dictionary with preprocessing settings
        
    Returns:
        DataFrame: Combined dataset with aligned timestamps
    """
    if config is None:
        config = {}
    
    # Convert grid_data to DataFrame
    grid_df = None
    if isinstance(grid_data, dict):
        if 'nodes' in grid_data and 'lines' in grid_data:
            # Extract nodes and lines and combine them
            nodes_df = grid_data.get('nodes')
            lines_df = grid_data.get('lines')
            
            # Convert to DataFrame if they're dictionaries
            if isinstance(nodes_df, dict):
                nodes_df = pd.DataFrame.from_dict(nodes_df, orient='index')
            if isinstance(lines_df, dict):
                lines_df = pd.DataFrame.from_dict(lines_df, orient='index')
                
            # Add component_type field
            if not nodes_df.empty:
                nodes_df['component_type'] = 'node'
            if not lines_df.empty:
                lines_df['component_type'] = 'line'
                
            # Combine into a single DataFrame
            grid_df = pd.concat([nodes_df, lines_df], ignore_index=True)
        else:
            # Maybe it's already a unified grid dataframe
            grid_df = pd.DataFrame(grid_data)
    else:
        # Assume it's already a DataFrame
        grid_df = grid_data
    
    # If we don't have valid grid data, return empty DataFrame
    if grid_df is None or grid_df.empty:
        return pd.DataFrame()
    
    # Create grid-centric timeline (one row per component per time period)
    
    # First, determine the time range from the weather and outage data
    start_time = min(
        weather_df['timestamp'].min(),
        outage_df['start_time'].min() if 'start_time' in outage_df.columns else pd.Timestamp.now()
    )
    
    end_time = max(
        weather_df['timestamp'].max(),
        outage_df['end_time'].max() if 'end_time' in outage_df.columns else pd.Timestamp.now()
    )
    
    # Create daily timeline
    timeline = pd.date_range(start=start_time, end=end_time, freq='D')
    
    # Create component-time combinations
    component_ids = grid_df['component_id'].unique()
    
    # Initialize combined dataframe
    combined_records = []
    
    # For each component, create timeline with weather and outage data
    for component_id in component_ids:
        component_data = grid_df[grid_df['component_id'] == component_id].iloc[0].to_dict()
        
        for timestamp in timeline:
            record = {
                'timestamp': timestamp,
                'component_id': component_id,
                'component_type': component_data.get('component_type', 'unknown'),
                'age': component_data.get('age', np.nan),
                'capacity': component_data.get('capacity', np.nan),
                'centrality': component_data.get('centrality', np.nan),
                'outage_occurred': False,
                'time_to_failure': np.nan
            }
            
            # Get weather data for this timestamp
            # Find closest weather station to this component
            if all(col in component_data for col in ['location_x', 'location_y']) and \
               all(col in weather_df.columns for col in ['latitude', 'longitude']):
                
                # Get component location
                comp_x = component_data.get('location_x')
                comp_y = component_data.get('location_y')
                
                if pd.notna(comp_x) and pd.notna(comp_y):
                    # Find weather stations with data for this day
                    day_weather = weather_df[
                        (weather_df['timestamp'].dt.date == timestamp.date())
                    ]
                    
                    if not day_weather.empty:
                        # Calculate distances to each station
                        day_weather['distance'] = day_weather.apply(
                            lambda row: ((row['latitude'] - comp_y)**2 + 
                                         (row['longitude'] - comp_x)**2)**0.5 
                                        if pd.notna(row['latitude']) and pd.notna(row['longitude'])
                                        else float('inf'),
                            axis=1
                        )
                        
                        # Get closest station's data
                        closest_station = day_weather.loc[day_weather['distance'].idxmin()]
                        
                        # Add weather features
                        record['temperature'] = closest_station.get('temperature', np.nan)
                        record['wind_speed'] = closest_station.get('wind_speed', np.nan)
                        record['precipitation'] = closest_station.get('precipitation', np.nan)
                        record['is_extreme_weather'] = (
                            closest_station.get('is_extreme_temperature', False) or
                            closest_station.get('is_extreme_wind', False) or
                            closest_station.get('is_extreme_precipitation', False)
                        )
            
            # Check for outages
            if 'start_time' in outage_df.columns and 'end_time' in outage_df.columns:
                # Find outages for this component on this day
                component_outages = outage_df[
                    (outage_df['component_id'] == component_id) &
                    (outage_df['start_time'].dt.date <= timestamp.date()) &
                    (outage_df['end_time'].dt.date >= timestamp.date())
                ]
                
                if not component_outages.empty:
                    record['outage_occurred'] = True
                
                # Find time to next failure
                future_outages = outage_df[
                    (outage_df['component_id'] == component_id) &
                    (outage_df['start_time'].dt.date > timestamp.date())
                ]
                
                if not future_outages.empty:
                    next_outage = future_outages.sort_values('start_time').iloc[0]
                    time_diff = (next_outage['start_time'] - timestamp).total_seconds() / 3600  # hours
                    record['time_to_failure'] = time_diff
            
            combined_records.append(record)
    
    # Create DataFrame from records
    combined_df = pd.DataFrame(combined_records)
    
    return combined_df
