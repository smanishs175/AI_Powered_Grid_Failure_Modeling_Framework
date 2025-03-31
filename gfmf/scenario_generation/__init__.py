"""
Scenario Generation Module.

This module generates realistic grid failure scenarios based on:
1. Normal operating conditions
2. Extreme weather events
3. Compound extreme events

It also models cascading failures and validates scenarios against historical data.
"""

from .scenario_generation_module import ScenarioGenerationModule

__all__ = ['ScenarioGenerationModule']
