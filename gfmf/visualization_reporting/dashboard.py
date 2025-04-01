"""
Dashboard component for the Grid Failure Modeling Framework.

This module provides classes and functions for creating interactive
dashboards for monitoring grid status, visualizing performance metrics,
and supporting operational decision-making.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import threading
import webbrowser


class Dashboard:
    """
    Class for creating and managing interactive dashboards.
    
    This class provides methods for creating different types of dashboards,
    including operational, vulnerability, and policy dashboards.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Dashboard class.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Dashboard component")
        
        # Set default configuration if not provided
        self.config = config or {}
        
        # Set dashboard parameters
        self.port = self.config.get('port', 8050)
        self.theme = self.config.get('theme', 'light')
        self.refresh_interval = self.config.get('refresh_interval', 300)  # seconds
        self.default_layout = self.config.get('default_layout', 'grid')
        self.max_items_per_page = self.config.get('max_items_per_page', 6)
        
        # Store active dashboards
        self.active_dashboards = {}
    
    def launch(self, dashboard_type='operational', auto_refresh=True, 
              refresh_interval=None, components=None, output_dir=None):
        """
        Launch an interactive dashboard.
        
        Args:
            dashboard_type (str): Type of dashboard ('operational', 'vulnerability', 'policy').
            auto_refresh (bool): Whether to auto-refresh the dashboard.
            refresh_interval (int, optional): Refresh interval in seconds.
            components (list, optional): List of dashboard components to include.
            output_dir (str, optional): Directory to save dashboard configuration.
            
        Returns:
            dict: Dictionary with dashboard metadata and URL.
        """
        # Use default refresh interval if not specified
        if refresh_interval is None:
            refresh_interval = self.refresh_interval
        
        # Set default components based on dashboard type if not provided
        if components is None:
            if dashboard_type == 'operational':
                components = ['grid_status', 'outage_metrics', 'weather_alerts']
            elif dashboard_type == 'vulnerability':
                components = ['vulnerability_map', 'component_risk', 'threat_assessment']
            elif dashboard_type == 'policy':
                components = ['policy_recommendations', 'resource_allocation', 'action_priority']
            else:
                components = ['grid_status', 'vulnerability_map', 'policy_recommendations']
        
        # Create the dashboard app
        app = dash.Dash(__name__, external_stylesheets=['/assets/stylesheet.css'])
        app.title = f'GFMF {dashboard_type.capitalize()} Dashboard'
        
        # Set theme colors
        if self.theme == 'dark':
            colors = {
                'background': '#111111',
                'text': '#FFFFFF',
                'grid': '#333333',
                'accent': '#007BFF'
            }
        else:  # 'light' theme
            colors = {
                'background': '#FFFFFF',
                'text': '#111111',
                'grid': '#EEEEEE',
                'accent': '#007BFF'
            }
        
        # Create dashboard layout based on components
        app.layout = self._create_dashboard_layout(dashboard_type, components, colors, auto_refresh, refresh_interval)
        
        # Add callback for data updates
        if auto_refresh:
            self._add_data_update_callbacks(app, dashboard_type, components, refresh_interval)
        
        # Generate a unique ID for this dashboard
        dashboard_id = f"{dashboard_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save dashboard configuration if output directory provided
        config_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            config_path = os.path.join(output_dir, f"{dashboard_id}_config.json")
            
            dashboard_config = {
                'id': dashboard_id,
                'type': dashboard_type,
                'components': components,
                'auto_refresh': auto_refresh,
                'refresh_interval': refresh_interval,
                'theme': self.theme,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                json.dump(dashboard_config, f, indent=2)
        
        # Run the dashboard in a separate thread
        port = self.port
        url = f"http://127.0.0.1:{port}/"
        
        def run_dashboard():
            app.run_server(debug=False, port=port)
        
        dashboard_thread = threading.Thread(target=run_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Open the browser automatically
        webbrowser.open(url)
        
        # Store in active dashboards
        self.active_dashboards[dashboard_id] = {
            'app': app,
            'thread': dashboard_thread,
            'url': url,
            'type': dashboard_type,
            'components': components
        }
        
        # Prepare result
        result = {
            'id': dashboard_id,
            'url': url,
            'type': dashboard_type,
            'components': components,
            'update_frequency': refresh_interval,
            'config_path': config_path,
            'export_options': ['png', 'pdf', 'csv'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Launched {dashboard_type} dashboard at {url}")
        
        return result
    
    def _create_dashboard_layout(self, dashboard_type, components, colors, auto_refresh, refresh_interval):
        """
        Create the dashboard layout based on components.
        
        Args:
            dashboard_type (str): Type of dashboard.
            components (list): List of dashboard components to include.
            colors (dict): Color scheme for the dashboard.
            auto_refresh (bool): Whether to auto-refresh the dashboard.
            refresh_interval (int): Refresh interval in seconds.
            
        Returns:
            dash.html.Div: The dashboard layout.
        """
        # Create header
        header = html.Div([
            html.H1(f"{dashboard_type.capitalize()} Dashboard", 
                   style={'color': colors['text'], 'textAlign': 'center'}),
            html.Div([
                html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      id='last-update-time'),
                html.P("Auto-refresh: " + ("Enabled" if auto_refresh else "Disabled"),
                      style={'marginLeft': '20px'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ], style={'padding': '10px', 'backgroundColor': colors['background']})
        
        # Create component panels based on requested components
        component_panels = []
        
        for component in components:
            if component == 'grid_status':
                panel = self._create_grid_status_panel(colors)
            elif component == 'vulnerability_map':
                panel = self._create_vulnerability_map_panel(colors)
            elif component == 'component_risk':
                panel = self._create_component_risk_panel(colors)
            elif component == 'threat_assessment':
                panel = self._create_threat_assessment_panel(colors)
            elif component == 'outage_metrics':
                panel = self._create_outage_metrics_panel(colors)
            elif component == 'weather_alerts':
                panel = self._create_weather_alerts_panel(colors)
            elif component == 'policy_recommendations':
                panel = self._create_policy_recommendations_panel(colors)
            elif component == 'resource_allocation':
                panel = self._create_resource_allocation_panel(colors)
            elif component == 'action_priority':
                panel = self._create_action_priority_panel(colors)
            else:
                self.logger.warning(f"Unknown component: {component}, skipping")
                continue
            
            component_panels.append(panel)
        
        # Create grid or flow layout based on configuration
        if self.default_layout == 'grid':
            # Create a grid layout with maximum items per row
            rows = []
            for i in range(0, len(component_panels), 2):
                row_panels = component_panels[i:i+2]
                row = html.Div(row_panels, style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'justifyContent': 'space-around',
                    'margin': '10px'
                })
                rows.append(row)
            
            components_layout = html.Div(rows, style={
                'backgroundColor': colors['background']
            })
        else:  # 'flow' layout
            components_layout = html.Div(component_panels, style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'space-around',
                'backgroundColor': colors['background']
            })
        
        # Create footer
        footer = html.Div([
            html.Hr(),
            html.P("Grid Failure Modeling Framework",
                  style={'textAlign': 'center', 'color': colors['text']})
        ], style={'padding': '10px', 'backgroundColor': colors['background']})
        
        # Combine all elements
        layout = html.Div([
            header,
            components_layout,
            footer,
            # Hidden div for storing data
            html.Div(id='data-store', style={'display': 'none'}),
            # Automatic refresh interval
            dcc.Interval(
                id='interval-component',
                interval=refresh_interval * 1000,  # Convert to milliseconds
                n_intervals=0
            ) if auto_refresh else html.Div()
        ], style={
            'backgroundColor': colors['background'],
            'color': colors['text'],
            'fontFamily': 'Arial, sans-serif',
            'minHeight': '100vh'
        })
        
        return layout
    
    def _add_data_update_callbacks(self, app, dashboard_type, components, refresh_interval):
        """
        Add callbacks for auto-refreshing data.
        
        Args:
            app (dash.Dash): Dash application.
            dashboard_type (str): Type of dashboard.
            components (list): List of dashboard components.
            refresh_interval (int): Refresh interval in seconds.
        """
        # Update last update time
        @app.callback(
            Output('last-update-time', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_time(n):
            return f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add component-specific update callbacks
        for component in components:
            if component == 'grid_status':
                self._add_grid_status_callback(app)
            elif component == 'vulnerability_map':
                self._add_vulnerability_map_callback(app)
            elif component == 'component_risk':
                self._add_component_risk_callback(app)
            elif component == 'threat_assessment':
                self._add_threat_assessment_callback(app)
            elif component == 'outage_metrics':
                self._add_outage_metrics_callback(app)
            elif component == 'weather_alerts':
                self._add_weather_alerts_callback(app)
            elif component == 'policy_recommendations':
                self._add_policy_recommendations_callback(app)
            elif component == 'resource_allocation':
                self._add_resource_allocation_callback(app)
            elif component == 'action_priority':
                self._add_action_priority_callback(app)
    
    def _create_grid_status_panel(self, colors):
        """Create a panel for grid status visualization."""
        return html.Div([
            html.H3("Grid Status", style={'textAlign': 'center', 'color': colors['text']}),
            dcc.Graph(id='grid-status-graph'),
            html.Div([
                html.P("Operational: ", style={'marginRight': '5px'}),
                html.Span("85%", id='operational-status', 
                         style={'color': 'green', 'fontWeight': 'bold'})
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            html.Div([
                html.P("At Risk: ", style={'marginRight': '5px'}),
                html.Span("10%", id='at-risk-status', 
                         style={'color': 'orange', 'fontWeight': 'bold'})
            ], style={'display': 'flex', 'justifyContent': 'center'}),
            html.Div([
                html.P("Failed: ", style={'marginRight': '5px'}),
                html.Span("5%", id='failed-status', 
                         style={'color': 'red', 'fontWeight': 'bold'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background']
        })
    
    def _create_vulnerability_map_panel(self, colors):
        """Create a panel for vulnerability map visualization."""
        return html.Div([
            html.H3("Vulnerability Map", style={'textAlign': 'center', 'color': colors['text']}),
            dcc.Graph(id='vulnerability-map-graph'),
            html.Div([
                html.P("Show Weather Overlay"),
                dcc.RadioItems(
                    id='weather-overlay-toggle',
                    options=[
                        {'label': 'On', 'value': 'on'},
                        {'label': 'Off', 'value': 'off'}
                    ],
                    value='off',
                    style={'display': 'flex', 'justifyContent': 'center'}
                )
            ], style={'textAlign': 'center'})
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background']
        })
    
    def _create_component_risk_panel(self, colors):
        """Create a panel for component risk visualization."""
        return html.Div([
            html.H3("Component Risk Analysis", style={'textAlign': 'center', 'color': colors['text']}),
            dcc.Graph(id='component-risk-graph'),
            html.Div([
                html.P("Component Type:"),
                dcc.Dropdown(
                    id='component-type-dropdown',
                    options=[
                        {'label': 'All Components', 'value': 'all'},
                        {'label': 'Generators', 'value': 'generator'},
                        {'label': 'Transformers', 'value': 'transformer'},
                        {'label': 'Transmission Lines', 'value': 'line'}
                    ],
                    value='all'
                )
            ], style={'width': '80%', 'margin': 'auto'})
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background']
        })
    
    def _create_threat_assessment_panel(self, colors):
        """Create a panel for threat assessment visualization."""
        return html.Div([
            html.H3("Threat Assessment", style={'textAlign': 'center', 'color': colors['text']}),
            dcc.Graph(id='threat-assessment-graph'),
            html.Div([
                html.P("Threat Type:"),
                dcc.Dropdown(
                    id='threat-type-dropdown',
                    options=[
                        {'label': 'All Threats', 'value': 'all'},
                        {'label': 'Weather', 'value': 'weather'},
                        {'label': 'Physical', 'value': 'physical'},
                        {'label': 'Cyber', 'value': 'cyber'}
                    ],
                    value='all'
                )
            ], style={'width': '80%', 'margin': 'auto'})
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background']
        })
    
    def _create_outage_metrics_panel(self, colors):
        """Create a panel for outage metrics visualization."""
        return html.Div([
            html.H3("Outage Metrics", style={'textAlign': 'center', 'color': colors['text']}),
            dcc.Graph(id='outage-metrics-graph'),
            html.Div([
                html.P("Time Range:"),
                dcc.RadioItems(
                    id='time-range-toggle',
                    options=[
                        {'label': '24 Hours', 'value': '24h'},
                        {'label': '7 Days', 'value': '7d'},
                        {'label': '30 Days', 'value': '30d'}
                    ],
                    value='24h',
                    style={'display': 'flex', 'justifyContent': 'space-around'}
                )
            ], style={'width': '80%', 'margin': 'auto'})
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background']
        })
    
    def _create_weather_alerts_panel(self, colors):
        """Create a panel for weather alerts."""
        return html.Div([
            html.H3("Weather Alerts", style={'textAlign': 'center', 'color': colors['text']}),
            html.Div(id='weather-alerts-container', children=[
                html.Div([
                    html.H4("Severe Thunderstorm Warning", style={'color': 'red'}),
                    html.P("Regions affected: Northeast Grid Sector"),
                    html.P("Potential impact: High winds may damage overhead lines"),
                    html.P("Estimated duration: 3 hours")
                ], style={'border': '1px solid red', 'borderRadius': '5px', 'padding': '10px', 'margin': '10px'}),
                html.Div([
                    html.H4("Heat Advisory", style={'color': 'orange'}),
                    html.P("Regions affected: Southern Grid Sector"),
                    html.P("Potential impact: Increased load due to cooling demand"),
                    html.P("Estimated duration: 48 hours")
                ], style={'border': '1px solid orange', 'borderRadius': '5px', 'padding': '10px', 'margin': '10px'})
            ])
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background'],
            'maxHeight': '500px',
            'overflow': 'auto'
        })
    
    def _create_policy_recommendations_panel(self, colors):
        """Create a panel for policy recommendations."""
        return html.Div([
            html.H3("Policy Recommendations", style={'textAlign': 'center', 'color': colors['text']}),
            html.Div(id='policy-recommendations-container', children=[
                html.Div([
                    html.H4("High Priority", style={'color': 'red'}),
                    html.P("Recommended Action: Reduce load on northeastern transmission lines"),
                    html.P("Benefit: Avoid cascading failure during severe weather"),
                    html.P("Confidence: 85%")
                ], style={'border': '1px solid red', 'borderRadius': '5px', 'padding': '10px', 'margin': '10px'}),
                html.Div([
                    html.H4("Medium Priority", style={'color': 'orange'}),
                    html.P("Recommended Action: Increase reserve generation capacity"),
                    html.P("Benefit: Handle increased cooling demand in southern sector"),
                    html.P("Confidence: 72%")
                ], style={'border': '1px solid orange', 'borderRadius': '5px', 'padding': '10px', 'margin': '10px'}),
                html.Div([
                    html.H4("Low Priority", style={'color': 'green'}),
                    html.P("Recommended Action: Schedule maintenance for western sector lines"),
                    html.P("Benefit: Prevent future outages due to aging infrastructure"),
                    html.P("Confidence: 91%")
                ], style={'border': '1px solid green', 'borderRadius': '5px', 'padding': '10px', 'margin': '10px'})
            ])
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background'],
            'maxHeight': '500px',
            'overflow': 'auto'
        })
    
    def _create_resource_allocation_panel(self, colors):
        """Create a panel for resource allocation visualization."""
        return html.Div([
            html.H3("Resource Allocation", style={'textAlign': 'center', 'color': colors['text']}),
            dcc.Graph(id='resource-allocation-graph'),
            html.Div([
                html.P("Resource Type:"),
                dcc.Dropdown(
                    id='resource-type-dropdown',
                    options=[
                        {'label': 'Repair Crews', 'value': 'crews'},
                        {'label': 'Backup Generation', 'value': 'backup'},
                        {'label': 'Spare Parts', 'value': 'parts'}
                    ],
                    value='crews'
                )
            ], style={'width': '80%', 'margin': 'auto'})
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background']
        })
    
    def _create_action_priority_panel(self, colors):
        """Create a panel for action priority visualization."""
        return html.Div([
            html.H3("Action Priority", style={'textAlign': 'center', 'color': colors['text']}),
            dcc.Graph(id='action-priority-graph')
        ], style={
            'width': '45%',
            'minWidth': '400px',
            'margin': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'backgroundColor': colors['background']
        })
    
    def _add_grid_status_callback(self, app):
        """Add callback for updating grid status visualization."""
        @app.callback(
            [Output('grid-status-graph', 'figure'),
             Output('operational-status', 'children'),
             Output('at-risk-status', 'children'),
             Output('failed-status', 'children')],
            Input('interval-component', 'n_intervals')
        )
        def update_grid_status(n):
            # In a real implementation, fetch data from previous modules
            # For now, generate mock data
            
            # Create mock data for grid status
            status_counts = {
                'Operational': np.random.randint(80, 90),
                'At Risk': np.random.randint(5, 15),
                'Failed': np.random.randint(1, 5)
            }
            
            total = sum(status_counts.values())
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                hole=.3,
                marker=dict(colors=['green', 'orange', 'red'])
            )])
            
            fig.update_layout(
                title='Component Status Distribution',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            # Calculate percentages
            operational_pct = f"{status_counts['Operational'] / total * 100:.1f}%"
            at_risk_pct = f"{status_counts['At Risk'] / total * 100:.1f}%"
            failed_pct = f"{status_counts['Failed'] / total * 100:.1f}%"
            
            return fig, operational_pct, at_risk_pct, failed_pct
    
    def _add_vulnerability_map_callback(self, app):
        """Add callback for updating vulnerability map visualization."""
        @app.callback(
            Output('vulnerability-map-graph', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('weather-overlay-toggle', 'value')]
        )
        def update_vulnerability_map(n, show_weather):
            # In a real implementation, fetch data from previous modules
            # For now, generate mock data
            
            # Generate grid data
            x = np.linspace(0, 10, 20)
            y = np.linspace(0, 10, 20)
            X, Y = np.meshgrid(x, y)
            
            # Generate vulnerability data
            Z = (np.sin(X) * np.cos(Y) + 1) / 2  # Values between 0 and 1
            
            # Create base heatmap
            fig = go.Figure()
            
            # Add vulnerability heatmap
            fig.add_trace(go.Heatmap(
                x=x, y=y, z=Z,
                colorscale='RdYlGn_r',  # Red (high) to Green (low)
                colorbar=dict(title='Vulnerability Score')
            ))
            
            # Add weather overlay if requested
            if show_weather == 'on':
                # Generate weather data (for example, precipitation)
                weather_z = np.exp(-((X - 5) ** 2 + (Y - 5) ** 2) / 5)
                
                fig.add_trace(go.Contour(
                    x=x, y=y, z=weather_z,
                    colorscale='Blues',
                    showscale=False,
                    line=dict(width=0.5),
                    contours=dict(coloring='lines'),
                    opacity=0.6
                ))
            
            # Add node markers for key components
            node_x = [2, 5, 8, 3, 7]
            node_y = [3, 7, 4, 8, 2]
            node_text = ['G1', 'T1', 'G2', 'T2', 'G3']
            node_colors = ['blue', 'orange', 'blue', 'orange', 'blue']
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(size=12, color=node_colors),
                text=node_text,
                textposition='top center'
            ))
            
            # Add lines representing transmission lines
            edge_x = [2, 5, None, 5, 8, None, 5, 3, None, 8, 7]
            edge_y = [3, 7, None, 7, 4, None, 7, 8, None, 4, 2]
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color='black', width=1.5),
                hoverinfo='none'
            ))
            
            fig.update_layout(
                title='Grid Vulnerability Map',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return fig
    
    # Following callbacks would be implemented similarly, generating mock data and updating
    # the respective visualization components
    
    def _add_component_risk_callback(self, app):
        """Add callback for updating component risk visualization."""
        pass  # Implementation similar to other callbacks
    
    def _add_threat_assessment_callback(self, app):
        """Add callback for updating threat assessment visualization."""
        pass  # Implementation similar to other callbacks
    
    def _add_outage_metrics_callback(self, app):
        """Add callback for updating outage metrics visualization."""
        pass  # Implementation similar to other callbacks
    
    def _add_weather_alerts_callback(self, app):
        """Add callback for updating weather alerts panel."""
        pass  # Implementation similar to other callbacks
    
    def _add_policy_recommendations_callback(self, app):
        """Add callback for updating policy recommendations panel."""
        pass  # Implementation similar to other callbacks
    
    def _add_resource_allocation_callback(self, app):
        """Add callback for updating resource allocation visualization."""
        pass  # Implementation similar to other callbacks
    
    def _add_action_priority_callback(self, app):
        """Add callback for updating action priority visualization."""
        pass  # Implementation similar to other callbacks
