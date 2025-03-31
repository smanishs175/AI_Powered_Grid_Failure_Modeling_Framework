#!/usr/bin/env python3
"""
Utah Grid Test Data Generator

This script generates synthetic but realistic grid data based on Utah's 
electrical infrastructure characteristics for testing the Grid Failure 
Modeling Framework.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random

# Set random seed for reproducibility
np.random.seed(2025)
random.seed(2025)

# Utah specific constants
UTAH_LAT_BOUNDS = (36.5, 42.0)  # Utah latitude bounds
UTAH_LON_BOUNDS = (-114.0, -109.0)  # Utah longitude bounds
UTAH_MAJOR_CITIES = [
    {"name": "Salt Lake City", "lat": 40.7608, "lon": -111.8910},
    {"name": "West Valley City", "lat": 40.6916, "lon": -112.0011},
    {"name": "Provo", "lat": 40.2338, "lon": -111.6585},
    {"name": "West Jordan", "lat": 40.6097, "lon": -112.0010},
    {"name": "Orem", "lat": 40.2969, "lon": -111.6946},
    {"name": "Sandy", "lat": 40.5649, "lon": -111.8389},
    {"name": "Ogden", "lat": 41.2230, "lon": -111.9738},
    {"name": "St. George", "lat": 37.0965, "lon": -113.5684},
    {"name": "Layton", "lat": 41.0602, "lon": -111.9710},
    {"name": "South Jordan", "lat": 40.5621, "lon": -111.9297}
]
UTAH_WEATHER_PATTERNS = {
    "winter": {
        "temperature": (-10, 5),  # Range in Celsius
        "precipitation": (0, 30),  # mm per day
        "humidity": (40, 80),      # percent
        "wind_speed": (0, 25)      # km/h
    },
    "spring": {
        "temperature": (5, 20),
        "precipitation": (10, 40),
        "humidity": (30, 70),
        "wind_speed": (5, 30)
    },
    "summer": {
        "temperature": (15, 35),
        "precipitation": (0, 20),
        "humidity": (10, 60),
        "wind_speed": (0, 20)
    },
    "fall": {
        "temperature": (5, 25),
        "precipitation": (5, 30),
        "humidity": (20, 65),
        "wind_speed": (0, 25)
    }
}


def generate_grid_topology():
    """Generate synthetic Utah grid topology."""
    
    # Define component types
    component_types = {
        "generating_station": {
            "count": 15,
            "capacity_range": (100, 1000),  # MW
            "age_range": (1, 40),  # years
            "near_cities": True
        },
        "transmission_substation": {
            "count": 30,
            "capacity_range": (300, 800),  # MVA
            "age_range": (1, 35),
            "near_cities": True
        },
        "distribution_substation": {
            "count": 60,
            "capacity_range": (50, 300),  # MVA
            "age_range": (1, 45),
            "near_cities": True
        },
        "transformer": {
            "count": 100,
            "capacity_range": (10, 100),  # MVA
            "age_range": (1, 50),
            "near_cities": False
        }
    }
    
    # Generate components
    components = []
    component_id = 0
    
    for comp_type, params in component_types.items():
        for i in range(params["count"]):
            if params["near_cities"] and i < len(UTAH_MAJOR_CITIES):
                # Place near a major city with some randomness
                city = UTAH_MAJOR_CITIES[i % len(UTAH_MAJOR_CITIES)]
                lat = city["lat"] + np.random.uniform(-0.1, 0.1)
                lon = city["lon"] + np.random.uniform(-0.1, 0.1)
            else:
                # Random location in Utah
                lat = np.random.uniform(UTAH_LAT_BOUNDS[0], UTAH_LAT_BOUNDS[1])
                lon = np.random.uniform(UTAH_LON_BOUNDS[0], UTAH_LON_BOUNDS[1])
            
            component = {
                "id": f"{comp_type}_{component_id}",
                "type": comp_type,
                "capacity": round(np.random.uniform(*params["capacity_range"]), 2),
                "age": round(np.random.uniform(*params["age_range"]), 1),
                "location": {"lat": lat, "lon": lon},
                "status": "operational",
                "maintenance_frequency": round(np.random.uniform(1, 12)),  # months
                "criticality": np.random.choice(["high", "medium", "low"], p=[0.2, 0.5, 0.3])
            }
            
            components.append(component)
            component_id += 1
    
    # Generate connections (transmission lines)
    connections = []
    
    # Connect generating stations to transmission substations
    gen_stations = [c for c in components if c["type"] == "generating_station"]
    trans_substations = [c for c in components if c["type"] == "transmission_substation"]
    
    for gen in gen_stations:
        # Connect each generating station to 1-3 closest transmission substations
        distances = [(calculate_distance(gen["location"], sub["location"]), sub) 
                    for sub in trans_substations]
        distances.sort(key=lambda x: x[0])
        
        num_connections = np.random.randint(1, min(4, len(distances)))
        for i in range(num_connections):
            _, sub = distances[i]
            connections.append({
                "id": f"transmission_line_{len(connections)}",
                "type": "transmission_line",
                "source": gen["id"],
                "target": sub["id"],
                "capacity": round(min(gen["capacity"], sub["capacity"]) * np.random.uniform(0.7, 1.0), 2),
                "length": round(distances[i][0], 2),  # km
                "age": round(np.random.uniform(1, 40), 1),
                "status": "operational",
                "voltage": np.random.choice([138, 230, 345, 500]),  # kV
                "maintenance_frequency": round(np.random.uniform(3, 24))  # months
            })
    
    # Connect transmission substations to distribution substations
    dist_substations = [c for c in components if c["type"] == "distribution_substation"]
    
    for trans in trans_substations:
        # Connect each transmission substation to 1-5 closest distribution substations
        distances = [(calculate_distance(trans["location"], sub["location"]), sub) 
                    for sub in dist_substations]
        distances.sort(key=lambda x: x[0])
        
        num_connections = np.random.randint(1, min(6, len(distances)))
        for i in range(num_connections):
            _, sub = distances[i]
            connections.append({
                "id": f"transmission_line_{len(connections)}",
                "type": "transmission_line",
                "source": trans["id"],
                "target": sub["id"],
                "capacity": round(min(trans["capacity"], sub["capacity"]) * np.random.uniform(0.7, 1.0), 2),
                "length": round(distances[i][0], 2),  # km
                "age": round(np.random.uniform(1, 40), 1),
                "status": "operational",
                "voltage": np.random.choice([69, 115, 138]),  # kV
                "maintenance_frequency": round(np.random.uniform(3, 24))  # months
            })
    
    # Connect distribution substations to transformers
    transformers = [c for c in components if c["type"] == "transformer"]
    
    for dist in dist_substations:
        # Connect each distribution substation to 1-3 closest transformers
        distances = [(calculate_distance(dist["location"], trans["location"]), trans) 
                    for trans in transformers]
        distances.sort(key=lambda x: x[0])
        
        num_connections = np.random.randint(1, min(4, len(distances)))
        for i in range(num_connections):
            _, trans = distances[i]
            connections.append({
                "id": f"distribution_line_{len(connections)}",
                "type": "distribution_line",
                "source": dist["id"],
                "target": trans["id"],
                "capacity": round(min(dist["capacity"], trans["capacity"]) * np.random.uniform(0.7, 1.0), 2),
                "length": round(distances[i][0], 2),  # km
                "age": round(np.random.uniform(1, 40), 1),
                "status": "operational",
                "voltage": np.random.choice([12.47, 13.8, 34.5]),  # kV
                "maintenance_frequency": round(np.random.uniform(6, 36))  # months
            })
    
    # Create the complete grid topology
    grid_topology = {
        "components": components,
        "connections": connections,
        "metadata": {
            "region": "Utah",
            "generated_on": datetime.now().isoformat(),
            "component_count": len(components),
            "connection_count": len(connections)
        }
    }
    
    return grid_topology


def calculate_distance(loc1, loc2):
    """Calculate approximate distance in km between two lat/lon points."""
    # Simple Euclidean distance - just for generating test data
    lat_diff = (loc1["lat"] - loc2["lat"]) * 111  # 1 degree lat ≈ 111 km
    lon_diff = (loc1["lon"] - loc2["lon"]) * 85   # 1 degree lon ≈ 85 km at Utah's latitude
    return np.sqrt(lat_diff**2 + lon_diff**2)


def generate_weather_data(start_date=None, days=365):
    """Generate synthetic weather data for Utah."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    
    # Generate daily weather data
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Determine season for each date
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    seasons = [get_season(date) for date in dates]
    
    # Generate weather parameters based on season
    weather_data = []
    
    for date, season in zip(dates, seasons):
        patterns = UTAH_WEATHER_PATTERNS[season]
        
        # Add random weather stations around Utah
        for city in UTAH_MAJOR_CITIES:
            # Add some randomness to weather parameters within seasonal ranges
            temperature = np.random.uniform(*patterns["temperature"])
            precipitation = np.random.uniform(*patterns["precipitation"])
            humidity = np.random.uniform(*patterns["humidity"])
            wind_speed = np.random.uniform(*patterns["wind_speed"])
            
            # Add small random variations for each city
            temperature += np.random.uniform(-3, 3)
            precipitation += np.random.uniform(-5, 5)
            humidity += np.random.uniform(-10, 10)
            wind_speed += np.random.uniform(-5, 5)
            
            # Ensure values are within reasonable bounds
            precipitation = max(0, precipitation)
            humidity = max(0, min(100, humidity))
            wind_speed = max(0, wind_speed)
            
            # Create weather record
            weather_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "station": city["name"],
                "latitude": city["lat"],
                "longitude": city["lon"],
                "temperature": round(temperature, 1),
                "precipitation": round(precipitation, 1),
                "humidity": round(humidity, 1),
                "wind_speed": round(wind_speed, 1),
                "season": season
            })
    
    # Convert to DataFrame
    weather_df = pd.DataFrame(weather_data)
    
    return weather_df


def generate_outage_data(grid_topology, weather_data, start_date=None, days=365):
    """Generate synthetic outage data based on grid topology and weather conditions."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    
    # Parameters that increase outage probability
    age_factor = 0.01  # % increase per year of age
    weather_thresholds = {
        "high_temp": 30,      # Celsius
        "low_temp": -5,       # Celsius
        "heavy_precipitation": 20,  # mm
        "high_wind": 15       # km/h
    }
    
    # Baseline outage probabilities by component type (daily)
    baseline_outage_prob = {
        "generating_station": 0.001,
        "transmission_substation": 0.002,
        "distribution_substation": 0.003,
        "transformer": 0.005,
        "transmission_line": 0.004,
        "distribution_line": 0.006
    }
    
    # Group weather data by date
    weather_by_date = {}
    for _, row in weather_data.iterrows():
        date = row["date"]
        if date not in weather_by_date:
            weather_by_date[date] = []
        weather_by_date[date].append(row)
    
    # Generate outages
    outages = []
    
    # Combined list of components and connections
    all_elements = grid_topology["components"] + grid_topology["connections"]
    
    # For each day in the period
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        current_date_str = current_date.strftime('%Y-%m-%d')
        
        # Skip if no weather data
        if current_date_str not in weather_by_date:
            continue
        
        # Get average weather conditions for this day
        day_weather = weather_by_date[current_date_str]
        avg_temp = np.mean([w["temperature"] for w in day_weather])
        max_precip = np.max([w["precipitation"] for w in day_weather])
        max_wind = np.max([w["wind_speed"] for w in day_weather])
        
        # Check each element for potential outage
        for element in all_elements:
            # Base probability from element type
            if element["type"] in baseline_outage_prob:
                base_prob = baseline_outage_prob[element["type"]]
            else:
                continue  # Skip if type not recognized
            
            # Adjust for age
            age_adjusted_prob = base_prob * (1 + age_factor * element["age"])
            
            # Adjust for weather
            weather_factor = 1.0
            
            if avg_temp > weather_thresholds["high_temp"]:
                weather_factor *= 1.5  # High temp increases failure probability
            
            if avg_temp < weather_thresholds["low_temp"]:
                weather_factor *= 2.0  # Very low temp increases failure probability significantly
            
            if max_precip > weather_thresholds["heavy_precipitation"]:
                weather_factor *= 2.5  # Heavy rain/snow increases failure probability
            
            if max_wind > weather_thresholds["high_wind"]:
                weather_factor *= 2.0  # High wind increases failure probability
            
            final_prob = age_adjusted_prob * weather_factor
            
            # Determine if outage occurs
            if np.random.random() < final_prob:
                # Outage occurs
                outage_duration = np.random.lognormal(mean=1.5, sigma=1.0)  # in hours
                outage_duration = min(max(1, outage_duration), 72)  # Between 1 and 72 hours
                
                start_time = current_date + timedelta(hours=np.random.randint(0, 24))
                end_time = start_time + timedelta(hours=outage_duration)
                
                # Determine cause
                if max_wind > weather_thresholds["high_wind"] and np.random.random() < 0.7:
                    cause = "high_wind"
                elif max_precip > weather_thresholds["heavy_precipitation"] and np.random.random() < 0.6:
                    cause = "heavy_precipitation"
                elif avg_temp < weather_thresholds["low_temp"] and np.random.random() < 0.5:
                    cause = "extreme_cold"
                elif avg_temp > weather_thresholds["high_temp"] and np.random.random() < 0.4:
                    cause = "extreme_heat"
                elif element["age"] > 30 and np.random.random() < 0.6:
                    cause = "equipment_aging"
                else:
                    cause = np.random.choice(["equipment_failure", "maintenance", "unknown", "animal_contact", "human_error"], 
                                            p=[0.5, 0.2, 0.1, 0.1, 0.1])
                
                # Create outage record
                outage = {
                    "component_id": element["id"],
                    "component_type": element["type"],
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_hours": round(outage_duration, 2),
                    "cause": cause,
                    "affected_capacity": round(element["capacity"] * np.random.uniform(0.1, 1.0), 2),
                    "weather_conditions": {
                        "temperature": round(avg_temp, 1),
                        "precipitation": round(max_precip, 1),
                        "wind_speed": round(max_wind, 1)
                    }
                }
                
                outages.append(outage)
    
    # Convert to DataFrame
    outage_df = pd.DataFrame(outages)
    
    return outage_df


def save_test_data(output_dir='test_data/utah_grid'):
    """Generate and save all test data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    print("Generating Utah grid topology...")
    grid_topo = generate_grid_topology()
    
    print("Generating weather data...")
    weather_data = generate_weather_data()
    
    print("Generating outage data...")
    outage_data = generate_outage_data(grid_topo, weather_data)
    
    # Save data to files
    print("Saving grid topology...")
    with open(os.path.join(output_dir, 'utah_grid_topology.json'), 'w') as f:
        json.dump(grid_topo, f, indent=2)
    
    print("Saving weather data...")
    weather_data.to_csv(os.path.join(output_dir, 'utah_weather_data.csv'), index=False)
    
    print("Saving outage data...")
    outage_data.to_csv(os.path.join(output_dir, 'utah_outage_data.csv'), index=False)
    
    print(f"All data saved to {output_dir}")
    print(f"Generated {len(grid_topo['components'])} components, {len(grid_topo['connections'])} connections")
    print(f"Generated {len(weather_data)} weather records")
    print(f"Generated {len(outage_data)} outage records")


if __name__ == "__main__":
    save_test_data()
