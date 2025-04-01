#!/usr/bin/env python
"""
Cascading Failure Model

This module models how component failures cascade through the grid network.
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

class CascadingFailureModel:
    """
    Models the cascade of failures through the grid network.
    
    This class analyzes how initial component failures can propagate through
    the grid network, causing additional failures.
    """
    
    def __init__(self, config=None):
        """
        Initialize the cascading failure model.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set default parameters
        self.max_cascade_steps = config.get('max_cascade_steps', 10)
        self.load_redistribution_factor = config.get('load_redistribution_factor', 0.6)
        self.capacity_threshold = config.get('capacity_threshold', 0.9)
        
        # Network model for different scenario types
        self.network_models = {}
        self.propagation_models = {}
    
    def model_cascading_failures(self, input_data, scenarios):
        """
        Model cascading failures for all scenarios.
        
        Args:
            input_data (dict): Dictionary containing input data.
            scenarios (dict): Dictionary mapping scenario types to lists of scenarios.
            
        Returns:
            dict: Dictionary containing cascade results for all scenarios.
        """
        self.logger.info("Modeling cascading failures for all scenarios")
        
        # Extract grid component data
        components = input_data.get('components', pd.DataFrame())
        
        if components.empty:
            self.logger.warning("No component data provided, cannot model cascading failures")
            return {
                'network_model': None,
                'propagation_models': {},
                'results': {}
            }
        
        # Build network model
        network_model = self._build_network_model(components)
        
        # Build propagation models for different scenario types
        propagation_models = self._build_propagation_models(network_model, scenarios)
        
        # Results container
        cascade_results = {}
        
        # Process each scenario type
        for scenario_type, scenario_list in scenarios.items():
            self.logger.info(f"Modeling cascading failures for {len(scenario_list)} {scenario_type} scenarios")
            
            # Apply the appropriate propagation model
            scenario_cascade_results = []
            
            for scenario in scenario_list:
                cascade = self._model_scenario_cascade(
                    scenario,
                    network_model,
                    propagation_models.get(scenario_type, propagation_models.get('normal', None))
                )
                
                # Store the result
                scenario_cascade_results.append({
                    'scenario_id': scenario['scenario_id'],
                    'cascade': cascade
                })
            
            cascade_results[scenario_type] = scenario_cascade_results
        
        # Compile and return all results
        return {
            'network_model': network_model,
            'propagation_models': propagation_models,
            'results': cascade_results
        }
    
    def _build_network_model(self, components):
        """
        Build a network model of the grid.
        
        Args:
            components (DataFrame): DataFrame containing component information.
            
        Returns:
            nx.Graph: NetworkX graph representing the grid.
        """
        self.logger.info("Building grid network model")
        
        # Create a directed graph to better represent power flow and dependencies
        G = nx.DiGraph()
        
        # Add components as nodes
        for _, component in components.iterrows():
            comp_id = component['component_id']
            G.add_node(
                comp_id,
                type=component.get('type', 'unknown'),
                capacity=component.get('capacity', 100.0),
                vulnerability=component.get('vulnerability_score', 0.5),
                age=component.get('age', 10),
                voltage=component.get('voltage', 0.0)
            )
        
        # Categorize components by type
        generators = []
        transformers = []
        substations = []
        transmission_lines = []
        distribution_lines = []
        other_components = []
        
        for node, attrs in G.nodes(data=True):
            node_type = attrs['type'].lower() if 'type' in attrs else ''
            
            if 'generator' in node_type or 'plant' in node_type:
                generators.append(node)
            elif 'transform' in node_type:
                transformers.append(node)
            elif 'substation' in node_type:
                substations.append(node)
            elif 'transmission' in node_type or 'line' in node_type:
                transmission_lines.append(node)
            elif 'distribution' in node_type:
                distribution_lines.append(node)
            else:
                other_components.append(node)
        
        # Connect components following a realistic grid hierarchy:
        # 1. Generators connect to transmission lines/substations
        self._connect_components(G, generators, transmission_lines + substations, p=0.6, min_connections=1, max_connections=3)
        
        # 2. Transmission lines connect to substations
        self._connect_components(G, transmission_lines, substations, p=0.7, min_connections=1, max_connections=4)
        
        # 3. Substations connect to transformers/distribution lines
        self._connect_components(G, substations, transformers + distribution_lines, p=0.8, min_connections=2, max_connections=6)
        
        # 4. Transformers connect to distribution lines
        self._connect_components(G, transformers, distribution_lines, p=0.7, min_connections=1, max_connections=3)
        
        # 5. Add connections between components of same type for redundancy
        self._connect_within_group(G, substations, p=0.2, bidirectional=True)
        self._connect_within_group(G, transmission_lines, p=0.1, bidirectional=True)
        
        # Ensure we have a connected graph - add some random connections if needed
        if nx.number_weakly_connected_components(G) > 1 and len(G.nodes) > 0:
            self._ensure_connectivity(G)
            
        # If we still have no edges at all, create a simple connected topology
        if len(G.edges) == 0 and len(G.nodes) > 1:
            all_nodes = list(G.nodes)
            for i in range(len(all_nodes)-1):
                G.add_edge(all_nodes[i], all_nodes[i+1], weight=0.9)
                
            # Add a few more random edges
            for _ in range(min(10, len(all_nodes))):
                src = np.random.choice(all_nodes)
                dst = np.random.choice(all_nodes)
                if src != dst and not G.has_edge(src, dst):
                    G.add_edge(src, dst, weight=np.random.uniform(0.6, 0.9))
        
        self.logger.info(f"Built network model with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _connect_components(self, G, source_nodes, target_nodes, p=0.5, min_connections=1, max_connections=3):
        """
        Connect source nodes to target nodes with directed edges.
        
        Args:
            G (nx.DiGraph): The network graph
            source_nodes (list): List of source node IDs
            target_nodes (list): List of target node IDs
            p (float): Base probability of making a connection
            min_connections (int): Minimum number of connections per source
            max_connections (int): Maximum number of connections per source
        """
        if not source_nodes or not target_nodes:
            return
            
        for source in source_nodes:
            # Get existing connections
            existing_connections = list(G.successors(source)) if source in G else []
            
            # Skip if already sufficiently connected
            if len(existing_connections) >= max_connections:
                continue
                
            # Potential targets (exclude already connected nodes)
            potential_targets = [t for t in target_nodes if t != source and t not in existing_connections]
            
            if not potential_targets:
                continue
                
            # Determine number of new connections for this source
            remaining_to_connect = max_connections - len(existing_connections)
            n_connections = min(remaining_to_connect, np.random.randint(min_connections, max_connections + 1))
            n_connections = min(n_connections, len(potential_targets))
            
            if n_connections <= 0:
                continue
                
            # Randomly select targets with higher probability based on node attributes
            weights = []
            for target in potential_targets:
                # Base probability
                weight = p
                
                # Adjust based on target attributes if available
                if 'voltage' in G.nodes[target] and 'voltage' in G.nodes[source]:
                    # Prefer connecting to similar voltage levels
                    source_voltage = float(G.nodes[source]['voltage'] or 0)
                    target_voltage = float(G.nodes[target]['voltage'] or 0)
                    
                    if source_voltage > 0 and target_voltage > 0:
                        voltage_ratio = min(source_voltage, target_voltage) / max(source_voltage, target_voltage)
                        weight *= (0.5 + 0.5 * voltage_ratio)  # Scale weight based on voltage similarity
                        
                # Adjust based on vulnerability (prefential attachment to less vulnerable components)
                if 'vulnerability' in G.nodes[target]:
                    # Slight preference for less vulnerable components
                    weight *= (1.0 - 0.2 * G.nodes[target]['vulnerability'])
                    
                weights.append(max(0.1, weight))  # Ensure minimum weight
                
            # Normalize weights
            if sum(weights) > 0:
                weights = [w/sum(weights) for w in weights]
            else:
                weights = None
                
            # Select targets based on weights
            selected_targets = np.random.choice(
                potential_targets, 
                size=min(n_connections, len(potential_targets)), 
                replace=False,
                p=weights
            )
            
            # Add edges
            for target in selected_targets:
                # Add edge with weight based on attributes
                base_weight = np.random.uniform(0.6, 0.9)
                
                # Adjust weight based on vulnerability (more vulnerable = higher cascade probability)
                if 'vulnerability' in G.nodes[source] and 'vulnerability' in G.nodes[target]:
                    src_vuln = G.nodes[source]['vulnerability']
                    tgt_vuln = G.nodes[target]['vulnerability']
                    vuln_factor = (src_vuln + tgt_vuln) / 2.0
                    
                    # Higher vulnerability means higher weight (more likely to fail)
                    weight_adj = base_weight * (1.0 + 0.2 * vuln_factor)
                else:
                    weight_adj = base_weight
                    
                G.add_edge(source, target, weight=min(0.95, weight_adj))

    def _connect_within_group(self, G, nodes, p=0.3, bidirectional=False):
        """
        Create connections between nodes in the same group.
        
        Args:
            G (nx.DiGraph): The network graph
            nodes (list): List of node IDs to connect within
            p (float): Probability of connecting any two nodes
            bidirectional (bool): Whether to create connections in both directions
        """
        if len(nodes) < 2:
            return
            
        for i, source in enumerate(nodes):
            for target in nodes[i+1:]:  # Only connect each pair once
                if np.random.random() < p:
                    # Create connection with random weight
                    weight = np.random.uniform(0.4, 0.7)  # Lower weights for same-type connections
                    G.add_edge(source, target, weight=weight)
                    
                    # Add reverse connection if bidirectional
                    if bidirectional:
                        rev_weight = np.random.uniform(0.4, 0.7)
                        G.add_edge(target, source, weight=rev_weight)
    
    def _ensure_connectivity(self, G):
        """
        Ensure the graph is weakly connected by adding edges between components.
        
        Args:
            G (nx.DiGraph): The network graph to make connected
        """
        # Identify weakly connected components
        components = list(nx.weakly_connected_components(G))
        
        if len(components) <= 1:
            return  # Already connected
            
        # Connect components by adding edges
        main_component = list(components[0])  # Start with largest component
        
        for component in components[1:]:
            # Find a random node from each component to connect
            source = np.random.choice(main_component)
            target = np.random.choice(list(component))
            
            # Connect with bidirectional edges
            G.add_edge(source, target, weight=0.8)
            G.add_edge(target, source, weight=0.8)
            
            # Add component to main component
            main_component.extend(component)
    
    def _build_propagation_models(self, network_model, scenarios):
        """
        Build cascade propagation models for different scenario types.
        
        Args:
            network_model (nx.Graph): NetworkX graph representing the grid.
            scenarios (dict): Dictionary mapping scenario types to lists of scenarios.
            
        Returns:
            dict: Dictionary mapping scenario types to propagation models.
        """
        self.logger.info("Building cascade propagation models")
        
        # Create models for each scenario type
        propagation_models = {}
        
        for scenario_type in scenarios.keys():
            # Default model parameters
            model_params = {
                'load_redistribution': self.load_redistribution_factor,
                'capacity_threshold': self.capacity_threshold,
                'step_factor': 1.0  # Base factor for step-wise propagation
            }
            
            # Adjust parameters based on scenario type
            if scenario_type == 'high_temperature':
                # High temperature reduces capacity
                model_params['capacity_threshold'] = self.capacity_threshold * 0.85
                model_params['step_factor'] = 1.2
            elif scenario_type == 'low_temperature':
                # Low temperature affects mechanical properties
                model_params['step_factor'] = 1.1
            elif scenario_type == 'high_wind':
                # High wind increases load redistribution issues
                model_params['load_redistribution'] = self.load_redistribution_factor * 1.3
                model_params['step_factor'] = 1.3
            elif scenario_type == 'precipitation':
                # Precipitation increases failure rate
                model_params['step_factor'] = 1.2
            elif scenario_type == 'compound':
                # Compound events are the worst
                model_params['capacity_threshold'] = self.capacity_threshold * 0.8
                model_params['load_redistribution'] = self.load_redistribution_factor * 1.4
                model_params['step_factor'] = 1.5
            
            propagation_models[scenario_type] = model_params
        
        return propagation_models
    
    def _model_scenario_cascade(self, scenario, network_model, propagation_model):
        """
        Model the cascade of failures for a single scenario.
        
        Args:
            scenario (dict): Scenario dictionary.
            network_model (nx.Graph): NetworkX graph representing the grid.
            propagation_model (dict): Propagation model parameters.
            
        Returns:
            dict: Cascade results.
        """
        # Get initial failed components
        initial_failures = list(scenario['component_failures'].keys())
        
        if not initial_failures:
            # No initial failures, no cascade
            return {
                'total_steps': 0,
                'final_failed_count': 0,
                'failed_components': [],
                'cascade_progression': []
            }
        
        # Copy the network model to avoid modifying the original
        cascade_network = network_model.copy()
        
        # Set up cascade simulation
        all_failed = set(initial_failures)
        newly_failed = set(initial_failures)
        cascade_steps = []
        
        # Add initial step
        cascade_steps.append({
            'step': 0,
            'newly_failed': list(newly_failed),
            'all_failed': list(all_failed)
        })
        
        # Initialize baseline load for all components (starting at proportion of capacity)
        for node in cascade_network.nodes():
            if 'capacity' in cascade_network.nodes[node]:
                capacity = cascade_network.nodes[node]['capacity']
                cascade_network.nodes[node]['baseline_load'] = capacity * 0.5  # Default 50% loading
                cascade_network.nodes[node]['current_load'] = capacity * 0.5
        
        # Define influence factors for different component types
        type_influence = {
            'generator': 1.5,      # Generators have high influence
            'plant': 1.5,          # Power plants have high influence
            'substation': 1.3,     # Substations have significant influence
            'transformer': 1.2,    # Transformers are important connection points
            'transmission': 1.1,   # Transmission lines affect multiple components
            'line': 1.0,           # Standard lines
            'distribution': 0.8    # Distribution lines affect fewer components
        }
        
        # Track failure causes for analysis
        failure_causes = {}
        for initial_fail in initial_failures:
            failure_causes[initial_fail] = 'initial'
        
        # Cascade simulation
        for step in range(1, self.max_cascade_steps + 1):
            # Get parameters for this step
            if propagation_model:
                load_redistribution = propagation_model.get('load_redistribution', self.load_redistribution_factor)
                capacity_threshold = propagation_model.get('capacity_threshold', self.capacity_threshold)
                step_factor = propagation_model.get('step_factor', 1.0)
            else:
                load_redistribution = self.load_redistribution_factor
                capacity_threshold = self.capacity_threshold
                step_factor = 1.0
            
            # Track components that fail in this step
            step_failures = set()
            step_failure_causes = {}
            
            # For each component that failed in the previous step
            for failed_comp in newly_failed:
                # Calculate load to redistribute based on component type and capacity
                if 'capacity' in cascade_network.nodes[failed_comp]:
                    base_load = cascade_network.nodes[failed_comp]['capacity']
                else:
                    base_load = 100.0  # Default capacity
                
                # Adjust load based on component type influence
                comp_type = cascade_network.nodes[failed_comp].get('type', '').lower()
                influence = 1.0
                for type_key, factor in type_influence.items():
                    if type_key in comp_type:
                        influence = factor
                        break
                
                load_to_redistribute = base_load * load_redistribution * influence
                
                # Identify downstream components (directed graph)
                downstream_comps = list(cascade_network.successors(failed_comp))
                
                # Also check for upstream components if they exist and add with lower weight
                upstream_comps = list(cascade_network.predecessors(failed_comp))
                upstream_weights = {}  # Store special weights for upstream
                
                # Calculate total edge weight for normalization
                total_weight = 0.0
                
                # Add downstream weights
                for comp in downstream_comps:
                    if comp not in all_failed:  # Skip already failed
                        edge_weight = cascade_network.get_edge_data(failed_comp, comp).get('weight', 1.0)
                        total_weight += edge_weight
                
                # Add upstream weights (reduced impact - backflow effect)
                for comp in upstream_comps:
                    if comp not in all_failed and comp not in downstream_comps:  # Skip if already counted
                        edge_weight = cascade_network.get_edge_data(comp, failed_comp).get('weight', 1.0) * 0.3  # Reduced impact
                        upstream_weights[comp] = edge_weight
                        total_weight += edge_weight
                
                # Combine all affected components
                affected_comps = [c for c in downstream_comps if c not in all_failed]
                affected_comps.extend([c for c in upstream_comps if c not in all_failed and c not in downstream_comps])
                
                if not affected_comps:
                    continue  # No components to affect
                
                # Distribute load to affected components
                for comp in affected_comps:
                    if comp in all_failed:
                        continue  # Skip already failed components
                    
                    # Get the appropriate edge weight
                    if comp in upstream_weights:  # It's an upstream component
                        edge_weight = upstream_weights[comp]
                    else:  # It's a downstream component
                        edge_weight = cascade_network.get_edge_data(failed_comp, comp).get('weight', 1.0)
                    
                    # Normalize and calculate load redistribution
                    if total_weight > 0:
                        load_proportion = edge_weight / total_weight
                    else:
                        load_proportion = 1.0 / len(affected_comps)
                    
                    redistributed_load = load_to_redistribute * load_proportion
                    
                    # Get component capacity
                    if 'capacity' in cascade_network.nodes[comp]:
                        comp_capacity = cascade_network.nodes[comp]['capacity']
                    else:
                        comp_capacity = 100.0  # Default
                    
                    # Update load
                    current_load = cascade_network.nodes[comp].get('current_load', comp_capacity * 0.5)
                    new_load = current_load + redistributed_load
                    cascade_network.nodes[comp]['current_load'] = new_load
                    
                    # Check for failure
                    # Higher load to capacity ratio increases failure probability
                    load_ratio = new_load / comp_capacity
                    
                    # Base failure threshold includes capacity threshold adjusted by scenario factors
                    failure_threshold = capacity_threshold * step_factor
                    
                    if load_ratio > failure_threshold:
                        # Calculate failure probability based on multiple factors
                        # 1. Load ratio (more overloaded = higher probability)
                        # 2. Component vulnerability
                        # 3. Step decay (cascades become less likely in later steps)
                        # 4. Random element
                        
                        # Load-based probability
                        load_prob = min(0.95, load_ratio / failure_threshold - 0.9)
                        
                        # Vulnerability factor
                        vulnerability = cascade_network.nodes[comp].get('vulnerability', 0.5)
                        vuln_factor = 0.7 + (vulnerability * 0.6)  # Scale to 0.7-1.3 range
                        
                        # Step decay - cascades become less likely in later steps
                        step_decay = 1.0 / (1.0 + (step * 0.15))  # Gradual decay
                        
                        # Calculate final probability
                        failure_prob = load_prob * vuln_factor * step_decay
                        
                        # Apply minimum probability based on step to ensure some progression
                        failure_prob = max(failure_prob, 0.05 / step)
                        
                        # Determine if component fails
                        if np.random.random() < failure_prob:
                            step_failures.add(comp)
                            step_failure_causes[comp] = failed_comp  # Track what caused this failure
            
            # Update tracking sets
            newly_failed = step_failures
            all_failed.update(newly_failed)
            
            # Update failure causes
            for comp, cause in step_failure_causes.items():
                failure_causes[comp] = cause
            
            # Collect detailed information about the failures in this step
            step_failure_details = []
            for comp in newly_failed:
                cause = step_failure_causes.get(comp, 'unknown')
                comp_type = cascade_network.nodes[comp].get('type', 'unknown')
                failure_detail = {
                    'component_id': comp,
                    'cause_id': cause,
                    'type': comp_type,
                    'load': cascade_network.nodes[comp].get('current_load', 0),
                    'capacity': cascade_network.nodes[comp].get('capacity', 100),
                    'vulnerability': cascade_network.nodes[comp].get('vulnerability', 0.5)
                }
                step_failure_details.append(failure_detail)
            
            # Record this step with detailed failure information
            cascade_steps.append({
                'step': step,
                'newly_failed': list(newly_failed),
                'all_failed': list(all_failed),
                'failure_details': step_failure_details
            })
            
            # Stop if no new failures
            if not newly_failed:
                break
        
        # Create failure chain data for visualization
        failure_chains = []
        for comp in all_failed:
            if comp in failure_causes:
                chain = [comp]
                current = comp
                
                # Trace back through the failure causes
                while current in failure_causes and failure_causes[current] != 'initial':
                    parent = failure_causes[current]
                    chain.append(parent)
                    current = parent
                
                # Reverse to get causal order (from initiator to final failure)
                chain.reverse()
                failure_chains.append({
                    'chain': chain,
                    'length': len(chain),
                    'initiator': chain[0] if chain else None
                })
        
        # Extract network data for visualization
        network_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add node data
        for node, attrs in cascade_network.nodes(data=True):
            node_data = {
                'id': node,
                'type': attrs.get('type', 'unknown'),
                'failed': node in all_failed,
                'load': attrs.get('current_load', 0),
                'capacity': attrs.get('capacity', 100),
                'vulnerability': attrs.get('vulnerability', 0.5)
            }
            # Add failure step information if applicable
            for i, step_data in enumerate(cascade_steps):
                if node in step_data['newly_failed']:
                    node_data['failure_step'] = i
                    break
            
            network_data['nodes'].append(node_data)
        
        # Add edge data
        for source, target, attrs in cascade_network.edges(data=True):
            edge_data = {
                'source': source,
                'target': target,
                'weight': attrs.get('weight', 1.0),
                'is_cascade_path': False
            }
            
            # Check if this edge is part of a failure chain
            for chain in failure_chains:
                chain_nodes = chain['chain']
                for i in range(len(chain_nodes)-1):
                    if chain_nodes[i] == source and chain_nodes[i+1] == target:
                        edge_data['is_cascade_path'] = True
                        break
            
            network_data['edges'].append(edge_data)
        
        # Compile enhanced results
        cascade_results = {
            'total_steps': len(cascade_steps) - 1,  # Exclude initial step
            'final_failed_count': len(all_failed),
            'failed_components': list(all_failed),
            'cascade_progression': cascade_steps,
            'failure_causes': {k: v for k, v in failure_causes.items()},
            'failure_chains': failure_chains,
            'network_data': network_data
        }
        
        return cascade_results
