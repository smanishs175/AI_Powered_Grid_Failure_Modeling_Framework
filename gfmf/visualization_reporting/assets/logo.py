"""
Script to generate a placeholder logo for the Grid Failure Modeling Framework.
This creates a simple logo image that can be used in reports and dashboards.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
import matplotlib.transforms as mtransforms
import numpy as np

def create_logo(output_path='logo.png', width=800, height=200, dpi=100):
    """
    Create a simple logo for the Grid Failure Modeling Framework.
    
    Args:
        output_path (str): Path where to save the logo.
        width (int): Width of the logo in pixels.
        height (int): Height of the logo in pixels.
        dpi (int): Resolution of the image.
    """
    # Calculate figure size in inches
    figsize = (width / dpi, height / dpi)
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Draw a stylized grid
    grid_color = '#0066cc'
    node_color = '#003366'
    highlight_color = '#e60000'
    
    # Create a grid network
    nodes = [
        (0.2, 0.5),
        (0.35, 0.7),
        (0.5, 0.3),
        (0.65, 0.6),
        (0.8, 0.4)
    ]
    
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)
    ]
    
    # Draw edges
    for i, j in edges:
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        if (i, j) == (2, 3):  # Highlight one edge for visual interest
            ax.plot([x1, x2], [y1, y2], color=highlight_color, linewidth=4, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], color=grid_color, linewidth=3, zorder=1)
    
    # Draw nodes
    for i, (x, y) in enumerate(nodes):
        if i == 2:  # Highlight one node for visual interest
            ax.scatter(x, y, s=300, color=highlight_color, zorder=2)
        else:
            ax.scatter(x, y, s=250, color=node_color, zorder=2)
    
    # Add text
    plt.text(0.5, 0.85, 'GFMF', 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=50, 
             weight='bold',
             color='#003366',
             zorder=3)
    
    plt.text(0.5, 0.15, 'Grid Failure Modeling Framework',
             horizontalalignment='center',
             verticalalignment='center', 
             fontsize=20,
             weight='bold',
             color='#333333',
             zorder=3)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Remove axis
    ax.axis('off')
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    
    print(f"Logo created at: {output_path}")

if __name__ == "__main__":
    # Create the logo in the assets directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'logo.png')
    create_logo(output_path=output_path)
