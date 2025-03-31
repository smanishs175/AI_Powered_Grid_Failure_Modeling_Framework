# Scenario Generation Components

This document describes the various scenario generation components in Module 4.

## Base Scenario Generator

The `BaseScenarioGenerator` class provides common functionality for all scenario generators, including:

- Generating unique scenario IDs
- Selecting components to fail based on failure probabilities
- Handling weather and environmental conditions
- Managing time-series data for scenario generation
- Providing a consistent interface for all generator types

## Normal Scenario Generator

The `NormalScenarioGenerator` class generates scenarios under normal operating conditions:

- Uses baseline failure probabilities from the Failure Prediction Module
- Applies minimal adjustments to reflect day-to-day variations
- Generates a distribution of failure scenarios that represent typical grid operations
- Includes normal weather conditions within expected ranges

## Extreme Event Scenario Generator

The `ExtremeEventScenarioGenerator` class generates scenarios during extreme weather events:

- Supports multiple extreme event types (high temperature, high wind, precipitation, low temperature)
- Applies event-specific impact models to adjust component failure probabilities
- Creates realistic weather conditions specific to each event type
- Models the temporal evolution of extreme events and their grid impacts

## Compound Scenario Generator

The `CompoundScenarioGenerator` class generates scenarios involving multiple concurrent extreme events:

- Combines multiple extreme events (e.g., high winds with precipitation)
- Models compound effects that may be greater than the sum of individual events
- Applies appropriate correlation between different weather factors
- Creates challenging but realistic failure scenarios for resilience testing

## Cascading Failure Model

The `CascadingFailureModel` class simulates how initial component failures can propagate through the grid:

- Builds a network representation of the grid infrastructure
- Models dependencies between components
- Simulates the step-by-step propagation of failures
- Calculates the final impact of cascading failures
- Identifies critical components and potential failure pathways

## Scenario Validator

The `ScenarioValidator` class ensures that generated scenarios are valid and useful:

- Checks for realism by comparing against historical data
- Ensures diversity in the generated scenario set
- Verifies physical consistency of weather conditions and component failures
- Calculates quality metrics for the scenario set
- Identifies and filters unrealistic or duplicate scenarios

Each of these components works together to create a comprehensive set of scenarios that can be used for grid vulnerability analysis, planning, and resilience testing.
