#!/usr/bin/env python
"""
Scenario Validator

This module validates generated scenarios against physical constraints
and historical data to ensure realism.
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import silhouette_score

class ScenarioValidator:
    """
    Validates scenarios for realism, diversity, and physical consistency.
    
    This class ensures that generated scenarios are realistic, diverse,
    and consistent with physical laws and historical data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the scenario validator.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Thresholds for validation metrics
        self.realism_threshold = config.get('realism_threshold', 0.7)
        self.diversity_threshold = config.get('diversity_threshold', 0.6)
        self.consistency_threshold = config.get('consistency_threshold', 0.8)
    
    def validate_scenarios(self, scenarios, historical_data=None):
        """
        Validate all generated scenarios.
        
        Args:
            scenarios (dict): Dictionary mapping scenario types to lists of scenarios.
            historical_data (DataFrame, optional): Historical outage data for validation.
            
        Returns:
            dict: Dictionary containing validation metrics.
        """
        self.logger.info("Validating generated scenarios")
        
        # Validate each scenario type
        type_metrics = {}
        all_scenarios = []
        
        for scenario_type, scenario_list in scenarios.items():
            self.logger.info(f"Validating {len(scenario_list)} {scenario_type} scenarios")
            
            # Calculate metrics for this scenario type
            realism_score = self._calculate_realism_score(scenario_list, historical_data)
            diversity_score = self._calculate_diversity_score(scenario_list)
            consistency_score = self._calculate_consistency_score(scenario_list, scenario_type)
            
            # Store metrics for this type
            type_metrics[scenario_type] = {
                'realism': realism_score,
                'diversity': diversity_score,
                'consistency': consistency_score
            }
            
            # Add to all scenarios for overall metrics
            all_scenarios.extend(scenario_list)
        
        # Calculate overall metrics
        overall_realism = self._calculate_realism_score(all_scenarios, historical_data)
        overall_diversity = self._calculate_diversity_score(all_scenarios)
        overall_consistency = self._calculate_consistency_score(all_scenarios)
        
        # Calculate severity distribution
        severity_distribution = self._calculate_severity_distribution(all_scenarios)
        
        # Calculate overall score (weighted average)
        overall_score = (
            overall_realism * 0.4 +
            overall_diversity * 0.3 +
            overall_consistency * 0.3
        )
        
        # Compile all validation metrics
        validation_metrics = {
            'realism_score': overall_realism,
            'diversity_score': overall_diversity,
            'consistency_score': overall_consistency,
            'severity_distribution': severity_distribution,
            'type_metrics': type_metrics,
            'overall_score': overall_score
        }
        
        self.logger.info(f"Validation complete: overall score {overall_score:.2f}")
        
        return validation_metrics
    
    def _calculate_realism_score(self, scenarios, historical_data=None):
        """
        Calculate realism score based on similarity to historical patterns.
        
        Args:
            scenarios (list): List of scenario dictionaries.
            historical_data (DataFrame, optional): Historical outage data.
            
        Returns:
            float: Realism score between 0 and 1.
        """
        if not scenarios:
            return 0.0
        
        # Start with a base score
        base_score = 0.7
        
        # If historical data is provided, compare against it
        if historical_data is not None and not historical_data.empty:
            try:
                # Compare failure patterns
                historical_components = set(historical_data['component_id'].unique())
                
                # Calculate overlap with historical components
                scenario_components = set()
                for scenario in scenarios:
                    scenario_components.update(scenario['component_failures'].keys())
                
                if historical_components and scenario_components:
                    component_overlap = len(scenario_components.intersection(historical_components)) / len(scenario_components)
                    
                    # Weather condition realism
                    weather_realism = self._check_weather_realism(scenarios, historical_data)
                    
                    # Combine into final score
                    realism_score = (component_overlap * 0.6) + (weather_realism * 0.4)
                    
                    # Ensure score is within [0,1]
                    return max(0, min(1, realism_score))
            except Exception as e:
                self.logger.warning(f"Error calculating realism against historical data: {e}")
                # Fall back to base score
                return base_score
        
        # If no historical data or comparison failed, use heuristics
        return self._calculate_heuristic_realism(scenarios)
    
    def _check_weather_realism(self, scenarios, historical_data):
        """
        Check if weather conditions in scenarios are realistic based on historical data.
        
        Args:
            scenarios (list): List of scenario dictionaries.
            historical_data (DataFrame): Historical weather and outage data.
            
        Returns:
            float: Weather realism score between 0 and 1.
        """
        # Simple heuristic check for now
        # In a real implementation, this would use statistical tests
        
        weather_realism = 0.75  # Default reasonable score
        
        # Check if the weather data has reasonable ranges
        valid_scenarios = 0
        for scenario in scenarios:
            weather = scenario['weather_conditions']
            
            # Get weather values and handle lists or tuples
            temperature = weather.get('temperature', 0)
            wind_speed = weather.get('wind_speed', 0)
            precipitation = weather.get('precipitation', 0)
            humidity = weather.get('humidity', 0)
            pressure = weather.get('pressure', 0)
            
            # Handle case where these might be lists or tuples
            if isinstance(temperature, (list, tuple)):
                temperature = sum(temperature) / len(temperature)  # Use average
            if isinstance(wind_speed, (list, tuple)):
                wind_speed = sum(wind_speed) / len(wind_speed)
            if isinstance(precipitation, (list, tuple)):
                precipitation = sum(precipitation) / len(precipitation)
            if isinstance(humidity, (list, tuple)):
                humidity = sum(humidity) / len(humidity)
            if isinstance(pressure, (list, tuple)):
                pressure = sum(pressure) / len(pressure)
            
            # Check if temperature is within reasonable bounds
            temp_valid = -50 <= temperature <= 50
            
            # Check if wind speed is within reasonable bounds
            wind_valid = 0 <= wind_speed <= 150
            
            # Check if precipitation is within reasonable bounds
            precip_valid = 0 <= precipitation <= 300
            
            # Check if humidity is within reasonable bounds
            humidity_valid = 0 <= humidity <= 100
            
            # Check if pressure is within reasonable bounds
            pressure_valid = 900 <= pressure <= 1100
            
            # Count valid scenarios
            if temp_valid and wind_valid and precip_valid and humidity_valid and pressure_valid:
                valid_scenarios += 1
        
        # Calculate proportion of valid scenarios
        if scenarios:
            weather_realism = valid_scenarios / len(scenarios)
        
        return weather_realism
    
    def _calculate_heuristic_realism(self, scenarios):
        """
        Calculate realism score based on heuristics.
        
        Args:
            scenarios (list): List of scenario dictionaries.
            
        Returns:
            float: Realism score between 0 and 1.
        """
        # Calculate average failures per scenario
        avg_failures = np.mean([len(s['component_failures']) for s in scenarios]) if scenarios else 0
        
        # Check if the average is reasonable (not too many or too few)
        if 1 <= avg_failures <= 20:
            failure_score = 0.8
        elif 0 < avg_failures < 1 or 20 < avg_failures <= 50:
            failure_score = 0.6
        else:
            failure_score = 0.4
        
        # Check weather patterns
        weather_score = self._check_weather_patterns(scenarios)
        
        # Check failure times distribution
        time_score = self._check_failure_time_distribution(scenarios)
        
        # Combine scores
        realism_score = (failure_score * 0.4) + (weather_score * 0.4) + (time_score * 0.2)
        
        return realism_score
    
    def _check_weather_patterns(self, scenarios):
        """
        Check if weather patterns in scenarios are realistic.
        
        Args:
            scenarios (list): List of scenario dictionaries.
            
        Returns:
            float: Weather pattern score between 0 and 1.
        """
        if not scenarios:
            return 0.5
        
        # Count scenarios with consistent weather conditions
        consistent_count = 0
        
        for scenario in scenarios:
            weather = scenario['weather_conditions']
            
            # Check for internal consistency
            is_consistent = True
            
            # Example consistency checks:
            # High temperature should not co-occur with cold snap
            if weather.get('temperature', 20) > 30 and weather.get('cold_snap_day', False):
                is_consistent = False
                
            # Low temperature should not co-occur with heat wave
            if weather.get('temperature', 20) < 5 and weather.get('heat_wave_day', False):
                is_consistent = False
                
            # High wind should be consistent with storm flag
            if weather.get('wind_speed', 10) > 40 and not weather.get('storm_day', False):
                is_consistent = False
                
            # High precipitation should be consistent with storm flag
            if weather.get('precipitation', 0) > 30 and not weather.get('storm_day', False):
                is_consistent = False
                
            # Count consistent scenarios
            if is_consistent:
                consistent_count += 1
        
        # Calculate proportion of consistent scenarios
        return consistent_count / len(scenarios)
    
    def _check_failure_time_distribution(self, scenarios):
        """
        Check if failure time distribution in scenarios is realistic.
        
        Args:
            scenarios (list): List of scenario dictionaries.
            
        Returns:
            float: Time distribution score between 0 and 1.
        """
        if not scenarios:
            return 0.5
        
        # Collect all failure times
        all_times = []
        
        for scenario in scenarios:
            for comp_id, failure in scenario['component_failures'].items():
                all_times.append(failure.get('failure_time', 0))
        
        if not all_times:
            return 0.5
        
        # Check if times are well distributed
        times = np.array(all_times)
        
        # Simple check for now: are the times spread out?
        # In a real implementation, this would use statistical tests
        
        # Calculate coefficient of variation (higher is more spread out)
        if np.mean(times) > 0:
            cv = np.std(times) / np.mean(times)
            
            # Convert to a score
            if cv > 0.5:
                return 0.8  # Good spread
            elif cv > 0.2:
                return 0.6  # Moderate spread
            else:
                return 0.4  # Poor spread
        
        return 0.5  # Default
    
    def _calculate_diversity_score(self, scenarios):
        """
        Calculate diversity score for scenarios.
        
        Args:
            scenarios (list): List of scenario dictionaries.
            
        Returns:
            float: Diversity score between 0 and 1.
        """
        if not scenarios or len(scenarios) < 2:
            return 0.0
        
        try:
            # Extract features for diversity calculation
            features = []
            
            for scenario in scenarios:
                # Number of failed components
                num_failures = len(scenario['component_failures'])
                
                # Average failure time
                failure_times = [f.get('failure_time', 0) for f in scenario['component_failures'].values()]
                avg_failure_time = np.mean(failure_times) if failure_times else 0
                
                # Weather features
                weather = scenario['weather_conditions']
                temperature = weather.get('temperature', 20)
                wind_speed = weather.get('wind_speed', 10)
                precipitation = weather.get('precipitation', 0)
                
                # Handle case where these might be lists or tuples
                if isinstance(temperature, (list, tuple)):
                    temperature = sum(temperature) / len(temperature)  # Use average
                if isinstance(wind_speed, (list, tuple)):
                    wind_speed = sum(wind_speed) / len(wind_speed)
                if isinstance(precipitation, (list, tuple)):
                    precipitation = sum(precipitation) / len(precipitation)
                
                # Create feature vector
                feature_vector = [num_failures, avg_failure_time, temperature, wind_speed, precipitation]
                features.append(feature_vector)
            
            # Convert to numpy array
            features = np.array(features)
            
            # Check if we have enough distinct data points
            if len(np.unique(features, axis=0)) < 2:
                return 0.5  # Not enough diversity to calculate
            
            # Feature normalization with safety checks
            means = np.mean(features, axis=0)
            stds = np.std(features, axis=0)
            
            # Replace zero standard deviations with 1 to avoid division by zero
            stds = np.where(stds > 0, stds, 1.0)
            
            # Standardize features
            features_std = (features - means) / stds
            features_std = np.nan_to_num(features_std)  # Handle any remaining NaNs
            
            # Calculate diversity using silhouette score if enough scenarios
            if len(scenarios) >= 4:
                try:
                    # Try to use silhouette score (needs at least 2 clusters)
                    from sklearn.cluster import KMeans
                    
                    # Determine optimal k
                    k = min(5, len(scenarios) // 2)
                    
                    # Fit KMeans
                    kmeans = KMeans(n_clusters=k).fit(features_std)
                    labels = kmeans.labels_
                    
                    # Calculate silhouette score
                    silhouette = silhouette_score(features_std, labels)
                    
                    # Convert to diversity score (higher silhouette is better)
                    diversity_score = (silhouette + 1) / 2  # Convert from [-1,1] to [0,1]
                    
                    return diversity_score
                except Exception as e:
                    self.logger.warning(f"Error calculating silhouette score: {e}")
            
            # Fall back to simpler diversity measure
            # Calculate average pairwise distance
            from scipy.spatial.distance import pdist
            
            pairwise_distances = pdist(features_std, 'euclidean')
            avg_distance = np.mean(pairwise_distances)
            
            # Normalize to [0,1] range (assuming distances are reasonably bounded)
            diversity_score = min(1.0, avg_distance / 5.0)
            
            return diversity_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating diversity: {e}")
            return 0.5  # Default moderate score
    
    def _calculate_consistency_score(self, scenarios, scenario_type=None):
        """
        Calculate physical consistency score for scenarios.
        
        Args:
            scenarios (list): List of scenario dictionaries.
            scenario_type (str, optional): Type of scenario for specific checks.
            
        Returns:
            float: Consistency score between 0 and 1.
        """
        if not scenarios:
            return 0.0
        
        # Track consistent scenarios
        consistent_scenarios = 0
        
        for scenario in scenarios:
            # Check weather-failure consistency
            weather_consistent = self._check_weather_failure_consistency(
                scenario['weather_conditions'],
                scenario['component_failures'],
                scenario_type
            )
            
            # Check timing consistency
            timing_consistent = self._check_timing_consistency(scenario['component_failures'])
            
            # Check metadata consistency
            metadata_consistent = True
            if scenario_type and 'metadata' in scenario:
                if scenario['metadata'].get('condition_type', '') != scenario_type:
                    metadata_consistent = False
            
            # Scenario is consistent if all checks pass
            if weather_consistent and timing_consistent and metadata_consistent:
                consistent_scenarios += 1
        
        # Calculate proportion of consistent scenarios
        consistency_score = consistent_scenarios / len(scenarios)
        
        return consistency_score
    
    def _check_weather_failure_consistency(self, weather, failures, scenario_type=None):
        """
        Check if failures are consistent with weather conditions.
        
        Args:
            weather (dict): Weather conditions.
            failures (dict): Component failures.
            scenario_type (str, optional): Type of scenario.
            
        Returns:
            bool: True if consistent, False otherwise.
        """
        # Default to consistent
        is_consistent = True
        
        # Get weather factors and ensure they are scalar values
        temperature = weather.get('temperature', 20)
        wind_speed = weather.get('wind_speed', 10)
        precipitation = weather.get('precipitation', 0)
        
        # Handle case where these might be lists or tuples (from configuration ranges)
        if isinstance(temperature, (list, tuple)):
            temperature = sum(temperature) / len(temperature)  # Use average
        if isinstance(wind_speed, (list, tuple)):
            wind_speed = sum(wind_speed) / len(wind_speed)
        if isinstance(precipitation, (list, tuple)):
            precipitation = sum(precipitation) / len(precipitation)
        
        # Check for inappropriate failure causes
        for comp_id, failure in failures.items():
            cause = failure.get('failure_cause', 'unknown')
            
            # Example consistency checks:
            
            # "Freezing" should not be a cause during high temperatures
            if cause == 'freezing' and temperature > 10:
                is_consistent = False
                
            # "Overheating" should not be a cause during low temperatures
            if cause == 'overheating' and temperature < 10:
                is_consistent = False
                
            # "Wind damage" should be associated with high winds
            if cause == 'wind_damage' and wind_speed < 20:
                is_consistent = False
                
            # "Water damage" should be associated with precipitation
            if cause == 'water_damage' and precipitation < 10:
                is_consistent = False
        
        # Additional checks based on scenario type
        if scenario_type:
            if scenario_type == 'high_temperature' and temperature < 30:
                is_consistent = False
                
            if scenario_type == 'low_temperature' and temperature > 0:
                is_consistent = False
                
            if scenario_type == 'high_wind' and wind_speed < 30:
                is_consistent = False
                
            if scenario_type == 'precipitation' and precipitation < 20:
                is_consistent = False
        
        return is_consistent
    
    def _check_timing_consistency(self, failures):
        """
        Check if failure timing is consistent.
        
        Args:
            failures (dict): Component failures.
            
        Returns:
            bool: True if consistent, False otherwise.
        """
        # Default to consistent
        is_consistent = True
        
        # Check that failure times are within reasonable bounds (e.g., 0-24 hours)
        for comp_id, failure in failures.items():
            time = failure.get('failure_time', 0)
            
            if time < 0 or time > 48:  # Assuming 48-hour scenario window
                is_consistent = False
        
        return is_consistent
    
    def _calculate_severity_distribution(self, scenarios):
        """
        Calculate the distribution of scenario severities.
        
        Args:
            scenarios (list): List of scenario dictionaries.
            
        Returns:
            dict: Counts of scenarios by severity level.
        """
        # Define severity levels based on number of failures
        severity_levels = {
            'low': 0,
            'medium': 0,
            'high': 0,
            'extreme': 0
        }
        
        for scenario in scenarios:
            num_failures = len(scenario['component_failures'])
            
            # Categorize by number of failures
            if num_failures <= 2:
                severity_levels['low'] += 1
            elif num_failures <= 5:
                severity_levels['medium'] += 1
            elif num_failures <= 10:
                severity_levels['high'] += 1
            else:
                severity_levels['extreme'] += 1
        
        return severity_levels
