import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from io import BytesIO
import base64
import logging
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

logger = logging.getLogger(__name__)

def create_visualization(viz_data):
    """
    Create visualizations based on the provided data and parameters
    Returns base64 encoded image
    """
    try:
        # Set up the plot
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        df = viz_data['data']
        x_col = viz_data['x_column']
        y_col = viz_data['y_column']
        
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Columns {x_col} or {y_col} not found in data")
        
        # Clean the data
        plot_data = df[[x_col, y_col]].dropna()
        
        # Create scatter plot
        if viz_data.get('type') == 'scatter' or 'scatter' in viz_data.get('title', '').lower():
            ax.scatter(plot_data[x_col], plot_data[y_col], 
                      alpha=0.7, 
                      color=viz_data.get('color', 'blue'),
                      s=50)
            
            # Add regression line if requested
            if viz_data.get('regression', False):
                try:
                    # Calculate regression
                    x_vals = plot_data[x_col].astype(float)
                    y_vals = plot_data[y_col].astype(float)
                    
                    # Remove any infinite or NaN values
                    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    x_clean = x_vals[mask]
                    y_clean = y_vals[mask]
                    
                    if len(x_clean) > 1:
                        # Fit regression line
                        coeffs = np.polyfit(x_clean, y_clean, 1)
                        poly_func = np.poly1d(coeffs)
                        
                        # Create line points
                        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                        y_line = poly_func(x_line)
                        
                        # Plot regression line
                        line_style = '--' if viz_data.get('regression_style') == 'dotted' else '-'
                        ax.plot(x_line, y_line, 
                               color=viz_data.get('regression_color', 'red'),
                               linestyle=line_style,
                               linewidth=2,
                               label=f'Regression line (slope: {coeffs[0]:.3f})')
                        
                        ax.legend()
                        
                except Exception as e:
                    logger.warning(f"Failed to add regression line: {e}")
        
        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(viz_data.get('title', f'{y_col} vs {x_col}'))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Check size (should be under 100KB)
        if len(image_base64) > 100000:
            logger.warning(f"Image size {len(image_base64)} bytes exceeds 100KB limit")
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        raise

def create_scatter_plot_with_regression(df, x_col, y_col, title="Scatter Plot"):
    """
    Create a scatter plot with regression line specifically for the evaluation criteria
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Clean the data
        plot_data = df[[x_col, y_col]].dropna()
        x_vals = pd.to_numeric(plot_data[x_col], errors='coerce')
        y_vals = pd.to_numeric(plot_data[y_col], errors='coerce')
        
        # Remove NaN values
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_clean = x_vals[mask]
        y_clean = y_vals[mask]
        
        if len(x_clean) < 2:
            raise ValueError("Not enough valid data points for plotting")
        
        # Create scatter plot
        ax.scatter(x_clean, y_clean, alpha=0.7, s=50, color='blue')
        
        # Add regression line (dotted red)
        coeffs = np.polyfit(x_clean, y_clean, 1)
        poly_func = np.poly1d(coeffs)
        
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = poly_func(x_line)
        
        ax.plot(x_line, y_line, color='red', linestyle=':', linewidth=2, label='Regression line')
        
        # Set labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Scatter plot creation failed: {e}")
        raise

def create_time_series_plot(df, x_col, y_col, title="Time Series"):
    """
    Create a time series plot
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Clean and sort data
        plot_data = df[[x_col, y_col]].dropna()
        plot_data = plot_data.sort_values(x_col)
        
        ax.plot(plot_data[x_col], plot_data[y_col], marker='o', linewidth=2, markersize=4)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if they're dates/strings
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Time series plot creation failed: {e}")
        raise
