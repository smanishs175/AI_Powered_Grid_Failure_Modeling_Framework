"""
Report generation component for the Grid Failure Modeling Framework.

This module provides functionality for generating automated reports
from visualization and analysis data.
"""

import os
import logging
from datetime import datetime
import jinja2
import matplotlib.pyplot as plt


class ReportGenerator:
    """
    Class for generating automated reports from visualization and analysis data.
    
    This class provides methods for creating different types of reports,
    including daily summaries, vulnerability assessments, and policy evaluation reports.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ReportGenerator class.
        
        Args:
            config (dict, optional): Configuration dictionary.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Report Generator component")
        
        # Set default configuration if not provided
        self.config = config or {}
        
        # Set report parameters
        self.template_dir = self.config.get('template_dir', 'templates/reports')
        self.default_format = self.config.get('default_format', 'pdf')
        self.logo_path = self.config.get('logo_path', 'assets/logo.png')
        self.company_name = self.config.get('company_name', 'Grid Resilience Inc.')
        self.default_sections = self.config.get('default_sections', 
                                              ['summary', 'vulnerability', 'predictions', 'policies'])
        
        # Initialize the template environment
        self._setup_template_environment()
    
    def _setup_template_environment(self):
        """
        Set up the template environment for report generation.
        """
        # Create template loader
        try:
            template_loader = jinja2.FileSystemLoader(searchpath=self.template_dir)
            self.template_env = jinja2.Environment(loader=template_loader)
            self.logger.info(f"Template environment set up with directory: {self.template_dir}")
        except Exception as e:
            self.logger.error(f"Error setting up template environment: {e}")
            self.logger.warning("Using default templates")
            # Use fallback templates from string
            self.template_env = jinja2.Environment()
            self._add_default_templates()
    
    def _add_default_templates(self):
        """
        Add default templates to the environment if no template directory is available.
        """
        # Default HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #003366; }
                h2 { color: #0055a4; border-bottom: 1px solid #ddd; }
                .summary { background-color: #f9f9f9; padding: 10px; }
                .vulnerability { background-color: #fff9f9; padding: 10px; }
                .predictions { background-color: #f9fff9; padding: 10px; }
                .policies { background-color: #f9f9ff; padding: 10px; }
                .footer { text-align: center; font-size: 0.8em; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>Generated on: {{ timestamp }}</p>
                <p>{{ company_name }}</p>
            </div>
            
            {% if 'summary' in sections %}
            <div class="summary">
                <h2>Executive Summary</h2>
                {{ summary_content }}
            </div>
            {% endif %}
            
            {% if 'vulnerability' in sections %}
            <div class="vulnerability">
                <h2>Vulnerability Assessment</h2>
                {{ vulnerability_content }}
                {% if vulnerability_image %}
                <img src="{{ vulnerability_image }}" alt="Vulnerability Map" style="max-width: 100%;">
                {% endif %}
            </div>
            {% endif %}
            
            {% if 'predictions' in sections %}
            <div class="predictions">
                <h2>Failure Predictions</h2>
                {{ predictions_content }}
                {% if predictions_image %}
                <img src="{{ predictions_image }}" alt="Failure Predictions" style="max-width: 100%;">
                {% endif %}
            </div>
            {% endif %}
            
            {% if 'policies' in sections %}
            <div class="policies">
                <h2>Policy Recommendations</h2>
                {{ policies_content }}
                {% if policies_image %}
                <img src="{{ policies_image }}" alt="Policy Performance" style="max-width: 100%;">
                {% endif %}
            </div>
            {% endif %}
            
            <div class="footer">
                <p>Copyright © {{ current_year }} {{ company_name }}</p>
            </div>
        </body>
        </html>
        """
        
        # Add templates to environment
        self.template_env.from_string(html_template)
    
    def generate_report(self, report_type='daily_summary', include_sections=None,
                       output_format=None, output_path=None):
        """
        Generate an automated report.
        
        Args:
            report_type (str): Type of report to generate.
            include_sections (list, optional): List of sections to include in the report.
            output_format (str, optional): Output format for the report.
            output_path (str, optional): Path where the report should be saved.
            
        Returns:
            dict: Dictionary with report metadata and file path.
        """
        self.logger.info(f"Generating {report_type} report")
        
        # Set default sections if not specified
        if include_sections is None:
            include_sections = self.default_sections
        
        # Set output format if not specified
        if output_format is None:
            output_format = self.default_format
        
        # Generate default output path if not specified
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_{timestamp}.{output_format}"
            output_path = os.path.join('outputs/visualization_reporting/reports', filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate report based on type
        if report_type == 'daily_summary':
            result = self._generate_daily_summary(include_sections, output_format, output_path)
        elif report_type == 'vulnerability_assessment':
            result = self._generate_vulnerability_assessment(include_sections, output_format, output_path)
        elif report_type == 'policy_evaluation':
            result = self._generate_policy_evaluation(include_sections, output_format, output_path)
        else:
            self.logger.warning(f"Unknown report type: {report_type}, using daily_summary")
            result = self._generate_daily_summary(include_sections, output_format, output_path)
        
        return result
    
    def _generate_daily_summary(self, include_sections, output_format, output_path):
        """
        Generate a daily summary report.
        
        Args:
            include_sections (list): List of sections to include in the report.
            output_format (str): Output format for the report.
            output_path (str): Path where the report should be saved.
            
        Returns:
            dict: Dictionary with report metadata and file path.
        """
        # In a real implementation, this would gather data from other modules
        # For now, we'll use placeholder content
        
        # Gather content for each section
        content = {
            'title': 'Daily Grid Status Summary',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'company_name': self.company_name,
            'current_year': datetime.now().year,
            'sections': include_sections,
            'summary_content': """
                <p>The grid is currently operating at 95% capacity with most components in operational status. 
                There are 3 components at risk and 1 component in failed status.</p>
                <p>Weather conditions pose moderate risk to northeastern sectors due to predicted thunderstorms.</p>
            """,
            'vulnerability_content': """
                <p>Component vulnerability analysis shows 5% of components with vulnerability scores above 0.7,
                primarily in the northeastern transmission lines.</p>
                <table>
                    <tr><th>Component Type</th><th>High Risk</th><th>Medium Risk</th><th>Low Risk</th></tr>
                    <tr><td>Generators</td><td>1</td><td>2</td><td>12</td></tr>
                    <tr><td>Transformers</td><td>0</td><td>3</td><td>9</td></tr>
                    <tr><td>Transmission Lines</td><td>4</td><td>7</td><td>32</td></tr>
                </table>
            """,
            'predictions_content': """
                <p>Failure prediction models indicate a 15% probability of at least one component failure
                in the next 24 hours, concentrated in the northeastern sector due to weather conditions.</p>
                <p>Expected load fluctuations due to temperature changes may impact southern generators.</p>
            """,
            'policies_content': """
                <p>Recommended actions from policy optimization:</p>
                <ol>
                    <li>Redistribute load from northeastern transmission lines to central sector.</li>
                    <li>Increase reserve capacity in southern generators to handle temperature-related demand.</li>
                    <li>Deploy maintenance crews to northeastern sector for preventative maintenance.</li>
                </ol>
            """
        }
        
        # Generate placeholder images for sections
        image_paths = {}
        if 'vulnerability' in include_sections:
            vulnerability_img_path = self._generate_placeholder_image('vulnerability_map', output_path)
            image_paths['vulnerability_image'] = vulnerability_img_path
            content['vulnerability_image'] = vulnerability_img_path
        
        if 'predictions' in include_sections:
            predictions_img_path = self._generate_placeholder_image('failure_predictions', output_path)
            image_paths['predictions_image'] = predictions_img_path
            content['predictions_image'] = predictions_img_path
        
        if 'policies' in include_sections:
            policies_img_path = self._generate_placeholder_image('policy_performance', output_path)
            image_paths['policies_image'] = policies_img_path
            content['policies_image'] = policies_img_path
        
        # Generate report based on output format
        if output_format.lower() == 'html':
            self._generate_html_report(content, output_path)
        elif output_format.lower() == 'pdf':
            self._generate_pdf_report(content, output_path)
        else:
            self.logger.warning(f"Unsupported output format: {output_format}, using HTML")
            output_path = output_path.replace(f".{output_format}", ".html")
            self._generate_html_report(content, output_path)
        
        # Prepare result
        result = {
            'report_type': 'daily_summary',
            'file_path': output_path,
            'sections': include_sections,
            'format': output_format,
            'timestamp': datetime.now().isoformat(),
            'image_paths': image_paths
        }
        
        return result
    
    def _generate_vulnerability_assessment(self, include_sections, output_format, output_path):
        """
        Generate a vulnerability assessment report.
        """
        # Implementation similar to _generate_daily_summary but with focus on vulnerability
        # For now, we'll just use the daily summary as placeholder
        return self._generate_daily_summary(include_sections, output_format, output_path)
    
    def _generate_policy_evaluation(self, include_sections, output_format, output_path):
        """
        Generate a policy evaluation report.
        """
        # Implementation similar to _generate_daily_summary but with focus on policies
        # For now, we'll just use the daily summary as placeholder
        return self._generate_daily_summary(include_sections, output_format, output_path)
    
    def _generate_placeholder_image(self, image_type, report_path):
        """
        Generate a placeholder image for the report.
        
        Args:
            image_type (str): Type of image to generate.
            report_path (str): Path where the report will be saved.
            
        Returns:
            str: Path to the generated image.
        """
        # Create figures directory
        figures_dir = os.path.join(os.path.dirname(report_path), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_type}_{timestamp}.png"
        image_path = os.path.join(figures_dir, filename)
        
        # Create a simple figure based on image type
        plt.figure(figsize=(10, 6))
        
        if image_type == 'vulnerability_map':
            # Create a simple heatmap
            data = [[0.2, 0.3, 0.8], [0.5, 0.4, 0.7], [0.1, 0.6, 0.3]]
            plt.imshow(data, cmap='RdYlGn_r')
            plt.colorbar(label='Vulnerability Score')
            plt.title('Grid Vulnerability Map')
            plt.grid(False)
        
        elif image_type == 'failure_predictions':
            # Create a simple line chart
            x = range(1, 8)
            y = [0.05, 0.07, 0.06, 0.15, 0.12, 0.08, 0.1]
            plt.plot(x, y, marker='o')
            plt.xlabel('Day')
            plt.ylabel('Failure Probability')
            plt.title('7-Day Failure Prediction')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        elif image_type == 'policy_performance':
            # Create a simple bar chart
            policies = ['Policy A', 'Policy B', 'Policy C', 'Policy D']
            performance = [0.82, 0.91, 0.75, 0.88]
            plt.bar(policies, performance)
            plt.xlabel('Policy')
            plt.ylabel('Performance Score')
            plt.title('Policy Performance Comparison')
            plt.ylim([0, 1])
        
        else:
            # Generic placeholder
            plt.text(0.5, 0.5, f"Placeholder for {image_type}", 
                   ha='center', va='center', fontsize=14)
            plt.axis('off')
        
        # Save the figure
        plt.savefig(image_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Return the relative path for HTML embedding
        return os.path.relpath(image_path, os.path.dirname(report_path))
    
    def _generate_html_report(self, content, output_path):
        """
        Generate an HTML report.
        
        Args:
            content (dict): Content for the report.
            output_path (str): Path where the report should be saved.
        """
        try:
            # Get HTML template
            template = self.template_env.from_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #003366; }
                    h2 { color: #0055a4; border-bottom: 1px solid #ddd; }
                    .summary { background-color: #f9f9f9; padding: 10px; }
                    .vulnerability { background-color: #fff9f9; padding: 10px; }
                    .predictions { background-color: #f9fff9; padding: 10px; }
                    .policies { background-color: #f9f9ff; padding: 10px; }
                    .footer { text-align: center; font-size: 0.8em; margin-top: 30px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>Generated on: {{ timestamp }}</p>
                    <p>{{ company_name }}</p>
                </div>
                
                {% if 'summary' in sections %}
                <div class="summary">
                    <h2>Executive Summary</h2>
                    {{ summary_content|safe }}
                </div>
                {% endif %}
                
                {% if 'vulnerability' in sections %}
                <div class="vulnerability">
                    <h2>Vulnerability Assessment</h2>
                    {{ vulnerability_content|safe }}
                    {% if vulnerability_image %}
                    <img src="{{ vulnerability_image }}" alt="Vulnerability Map" style="max-width: 100%;">
                    {% endif %}
                </div>
                {% endif %}
                
                {% if 'predictions' in sections %}
                <div class="predictions">
                    <h2>Failure Predictions</h2>
                    {{ predictions_content|safe }}
                    {% if predictions_image %}
                    <img src="{{ predictions_image }}" alt="Failure Predictions" style="max-width: 100%;">
                    {% endif %}
                </div>
                {% endif %}
                
                {% if 'policies' in sections %}
                <div class="policies">
                    <h2>Policy Recommendations</h2>
                    {{ policies_content|safe }}
                    {% if policies_image %}
                    <img src="{{ policies_image }}" alt="Policy Performance" style="max-width: 100%;">
                    {% endif %}
                </div>
                {% endif %}
                
                <div class="footer">
                    <p>Copyright © {{ current_year }} {{ company_name }}</p>
                </div>
            </body>
            </html>
            """)
            
            # Render template with content
            html_content = template.render(**content)
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report saved to: {output_path}")
        
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            raise
    
    def _generate_pdf_report(self, content, output_path):
        """
        Generate a PDF report.
        
        Args:
            content (dict): Content for the report.
            output_path (str): Path where the report should be saved.
        """
        try:
            # For now, we'll generate an HTML file with a .pdf extension
            # In a real implementation, this would use a PDF generation library like weasyprint
            html_path = output_path.replace('.pdf', '.html')
            self._generate_html_report(content, html_path)
            
            self.logger.warning("PDF generation not implemented yet, created HTML file instead")
            self.logger.info(f"HTML report (instead of PDF) saved to: {html_path}")
            
            # Update output path to the HTML file
            output_path = html_path
        
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            raise
        
        return output_path
