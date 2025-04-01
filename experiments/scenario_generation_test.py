#!/usr/bin/env python
"""
Scenario Generation Module - Integration Test Script

This script tests the Scenario Generation Module to ensure it works correctly
with inputs from the Failure Prediction Module.
"""

import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import scenario generation module
from gfmf.scenario_generation import ScenarioGenerationModule

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scenario_generation_test.log')
    ]
)
logger = logging.getLogger('ScenarioGenerationTest')

def main():
    """
    Main function to test the Scenario Generation Module.
    """
    logger.info("Starting Scenario Generation Module Integration Test")
    
    # Create output directory
    output_dir = Path("outputs/scenario_generation_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the module with default configuration
    logger.info("Initializing Scenario Generation Module")
    scenario_module = ScenarioGenerationModule()
    
    try:
        # Run scenario generation with synthetic data
        logger.info("Running scenario generation with synthetic data")
        generation_results = scenario_module.generate_scenarios(use_synthetic=True)
        
        # Access and print summary of results
        normal_scenarios = generation_results['normal_scenarios']
        extreme_scenarios = generation_results['extreme_scenarios']
        cascade_results = generation_results['cascade_results']
        validation_results = generation_results['validation_results']
        
        # Print normal scenario summary
        logger.info(f"Generated {len(normal_scenarios)} normal scenarios")
        
        # Print extreme scenario counts
        for event_type, scenarios in extreme_scenarios.items():
            if event_type != 'compound':
                logger.info(f"Generated {len(scenarios)} {event_type} scenarios")
        
        # Print compound scenario count
        compound_count = len(extreme_scenarios.get('compound', []))
        logger.info(f"Generated {compound_count} compound scenarios")
        
        # Print cascade statistics
        total_cascades = 0
        max_cascade_steps = 0
        
        for scenario_type, scenarios in cascade_results.get('results', {}).items():
            total_cascades += len(scenarios)
            for scenario in scenarios:
                cascade = scenario.get('cascade', {})
                steps = cascade.get('total_steps', 0)
                max_cascade_steps = max(max_cascade_steps, steps)
        
        logger.info(f"Modeled {total_cascades} cascading failure scenarios")
        logger.info(f"Maximum cascade propagation: {max_cascade_steps} steps")
        
        # Print validation scores
        logger.info(f"Overall scenario realism score: {validation_results['realism_score']:.2f}")
        logger.info(f"Scenario diversity score: {validation_results['diversity_score']:.2f}")
        logger.info(f"Physical consistency score: {validation_results['consistency_score']:.2f}")
        
        # Print severity distribution
        severity = validation_results.get('severity_distribution', {})
        logger.info("Scenario severity distribution:")
        for level, count in severity.items():
            logger.info(f"  {level}: {count} scenarios")
        
        # Save some examples to output directory
        logger.info("Saving example scenarios to output directory")
        
        # Example normal scenario
        if normal_scenarios:
            with open(output_dir / "example_normal_scenario.pkl", 'wb') as f:
                pickle.dump(normal_scenarios[0], f)
        
        # Example extreme scenarios
        for event_type, scenarios in extreme_scenarios.items():
            if scenarios:
                with open(output_dir / f"example_{event_type}_scenario.pkl", 'wb') as f:
                    pickle.dump(scenarios[0], f)
        
        # Visualize some results
        logger.info("Generating visualizations")
        
        # Scenario counts by type
        scenario_counts = {
            'normal': len(normal_scenarios),
            **{event_type: len(scenarios) for event_type, scenarios in extreme_scenarios.items()}
        }
        
        plt.figure(figsize=(10, 6))
        plt.bar(scenario_counts.keys(), scenario_counts.values())
        plt.title('Scenario Counts by Type')
        plt.xlabel('Scenario Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "scenario_counts.png")
        
        # Severity distribution
        plt.figure(figsize=(8, 6))
        plt.bar(severity.keys(), severity.values())
        plt.title('Scenario Severity Distribution')
        plt.xlabel('Severity Level')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / "severity_distribution.png")
        
        # Check cascade progression for a sample scenario
        cascade_sample = None
        
        # Find a scenario with a cascade
        for scenario_type, scenarios in cascade_results.get('results', {}).items():
            for scenario in scenarios:
                cascade = scenario.get('cascade', {})
                if cascade.get('total_steps', 0) > 1:
                    cascade_sample = cascade
                    break
            if cascade_sample:
                break
        
        if cascade_sample:
            # Extract data for visualization
            steps = [step['step'] for step in cascade_sample['cascade_progression']]
            failure_counts = [len(step['all_failed']) for step in cascade_sample['cascade_progression']]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps, failure_counts, 'o-', linewidth=2)
            plt.title('Cascade Progression Example')
            plt.xlabel('Cascade Step')
            plt.ylabel('Total Failed Components')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "cascade_progression.png")
        
        logger.info(f"Test complete! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in scenario generation test: {e}")
        import traceback
        traceback.print_exc()
        
    logger.info("Scenario Generation Module Integration Test completed!")

if __name__ == "__main__":
    main()
