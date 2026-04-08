import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Visualization generation for Blog 16: Mine-Site Thermal Anomaly Detection with MODIS
Creates minimalist-style visualizations for distributed thermal anomaly detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

# Import Tufte plotting utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors


warnings.filterwarnings('ignore')

def apply_minimalist_style_manual(ax):
    """Apply minimalist style components manually to axis."""
    plt.rcParams["font.family"] = "serif"
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
def generate_mine_thermal_data():
    """
    Generate synthetic MODIS LST data for mine sites with anomalies.
    """
    np.random.seed(42)
    
    # 365 days of data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
    days = np.arange(365)
    
    # Seasonal baseline temperature (Kelvin)
    baseline = 298 + 15 * np.sin((days - 80) * 2 * np.pi / 365)
    
    # Add daily variation
    baseline += np.random.randn(365) * 2.5
    
    # Normal mine site (follows baseline closely)
    normal_site = baseline + np.random.randn(365) * 1.5
    
    # Tailings dam with thermal anomalies (spontaneous combustion events)
    tailings_site = baseline.copy()
    # Add anomaly events
    anomaly_periods = [(120, 145), (220, 235), (310, 330)]
    for start, end in anomaly_periods:
        intensity = np.random.uniform(15, 25)
        tailings_site[start:end] += intensity * np.exp(-((np.arange(end-start) - (end-start)/2)**2) / 20)
    tailings_site += np.random.randn(365) * 2.0
    
    # Waste dump with gradual heating (oxidation)
    waste_site = baseline + 0.03 * days + np.random.randn(365) * 2.0
    
    return dates, baseline, normal_site, tailings_site, waste_site

def create_main_thermal_time_series():
    """
    Create time series plot showing thermal anomalies across mine sites.
    """
    print("Generating main thermal time series visualization...")
    
    dates, baseline, normal, tailings, waste = generate_mine_thermal_data()
    
    # Convert Kelvin to Celsius
    baseline_c = baseline - 273.15
    normal_c = normal - 273.15
    tailings_c = tailings - 273.15
    waste_c = waste - 273.15
    
    # Calculate z-scores for tailings
    z_scores = (tailings - baseline) / np.std(tailings - baseline)
    anomalies = np.abs(z_scores) > 3
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top panel: Temperature time series
    ax1.plot(dates, baseline_c, '--', color='black', linewidth=2, 
            label='Seasonal Baseline', alpha=0.7)
    
    ax1.plot(dates, normal_c, '-', color='black', linewidth=1.5, 
            label='Normal Mine Site', alpha=0.8)
    
    ax1.plot(dates, waste_c, '-', color='black', linewidth=1.5,
            label='Waste Dump (Oxidation)', alpha=0.8)
    
    ax1.plot(dates, tailings_c, '-', color='black', linewidth=1.5,
            label='Tailings Dam', alpha=0.8)
    
    # Highlight anomalies
    anomaly_dates = [d for d, a in zip(dates, anomalies) if a]
    anomaly_temps = [t for t, a in zip(tailings_c, anomalies) if a]
    
    ax1.scatter(anomaly_dates, anomaly_temps, s=100, 
               facecolors='none', edgecolors='#FF4136', linewidths=2,
               label=f'Thermal Anomalies (n={len(anomaly_dates)})', zorder=5)
    
    apply_minimalist_style_manual(ax1)
    ax1.set_ylabel('Temperature (°C)', fontsize=11)
    ax1.set_title('Mine-Site Thermal Monitoring with MODIS LST', 
                 fontsize=13, fontweight='bold', loc='left', pad=20)
    ax1.legend(loc='upper left', frameon=False, fontsize=9, ncol=2)
    
    # Add annotation for spontaneous combustion event
    event_idx = 128
    ax1.annotate('Spontaneous Combustion Event', 
                xy=(dates[event_idx], tailings_c[event_idx]), 
                xytext=(dates[event_idx + 50], tailings_c[event_idx] + 8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', 
                                     edgecolor='black', linewidth=1))
    
    # Bottom panel: Z-score anomaly detection
    ax2.plot(dates, z_scores, 'o-', color='black', linewidth=1, 
            markersize=2, markerfacecolor='white', markeredgecolor='black',
            label='Z-Score (Tailings Dam)')
    
    # Threshold lines
    ax2.axhline(y=3, color='black', linestyle='--', linewidth=2, label='Anomaly Threshold (±3σ)')
    ax2.axhline(y=-3, color='black', linestyle='--', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    # Shade anomaly regions
    ax2.fill_between(dates, -10, 10, where=np.abs(z_scores) > 3, 
                    color='black', alpha=0.2, label='Anomaly Regions')
    
    apply_minimalist_style_manual(ax2)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Z-Score (σ)', fontsize=11)
    ax2.set_title('Statistical Anomaly Detection', 
                 fontsize=12, fontweight='bold', loc='left', pad=15)
    ax2.legend(loc='upper left', frameon=False, fontsize=9)
    ax2.set_ylim(-6, 8)
    
    # Format x-axis
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/16_thermal_anomaly_modis_main.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Main thermal time series visualization saved")
    print(f"  Anomalies detected: {len(anomaly_dates)}")

def create_spatial_thermal_heatmap():
    """
    Create spatial heatmap showing thermal patterns across mine site.
    """
    print("Generating spatial thermal heatmap visualization...")
    
    np.random.seed(42)
    
    # Create spatial grid (10km x 10km mine site)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Baseline temperature field
    temp = 25 + 5 * np.sin(X * 0.5) + 3 * np.cos(Y * 0.7)
    
    # Add thermal anomalies at specific locations
    
    # Tailings dam (spontaneous combustion)
    tailings_x, tailings_y = 3.5, 7.0
    dist_tailings = np.sqrt((X - tailings_x)**2 + (Y - tailings_y)**2)
    temp += 20 * np.exp(-dist_tailings**2 / 0.4)
    
    # Waste dump (oxidation)
    waste_x, waste_y = 7.5, 3.5
    dist_waste = np.sqrt((X - waste_x)**2 + (Y - waste_y)**2)
    temp += 12 * np.exp(-dist_waste**2 / 0.8)
    
    # Processing plant (normal operations)
    plant_x, plant_y = 5.0, 5.0
    dist_plant = np.sqrt((X - plant_x)**2 + (Y - plant_y)**2)
    temp += 8 * np.exp(-dist_plant**2 / 0.3)
    
    # Add noise
    temp += np.random.randn(100, 100) * 1.5
    
    # Calculate z-scores
    z_scores = (temp - np.mean(temp)) / np.std(temp)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Temperature heatmap
    im1 = ax1.contourf(X, Y, temp, levels=20, cmap='hot')
    
    # Add mine features
    ax1.plot(tailings_x, tailings_y, 'c^', markersize=15, 
            markeredgecolor='black', markeredgewidth=2, label='Tailings Dam')
    ax1.plot(waste_x, waste_y, 'cs', markersize=15,
            markeredgecolor='black', markeredgewidth=2, label='Waste Dump')
    ax1.plot(plant_x, plant_y, 'co', markersize=15,
            markeredgecolor='black', markeredgewidth=2, label='Processing Plant')
    
    apply_minimalist_style_manual(ax1)
    ax1.set_xlabel('Easting (km)', fontsize=10)
    ax1.set_ylabel('Northing (km)', fontsize=10)
    ax1.set_title('MODIS LST Temperature Map', 
                  fontsize=12, fontweight='bold', loc='center', pad=15)
    ax1.legend(loc='upper right', frameon=False, fontsize=8)
    ax1.set_aspect('equal')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Temperature (°C)', fontsize=10)
    cbar1.outline.set_visible(False)
    
    # Right panel: Z-score anomaly map
    im2 = ax2.contourf(X, Y, z_scores, levels=20, cmap='gray')
    
    # Highlight high anomalies (>3σ)
    ax2.contour(X, Y, z_scores, levels=[3], colors='red', linewidths=3, linestyles='--')
    
    # Add mine features
    ax2.plot(tailings_x, tailings_y, 'k^', markersize=15, 
            markeredgecolor='white', markeredgewidth=2, label='Tailings Dam')
    ax2.plot(waste_x, waste_y, 'ks', markersize=15,
            markeredgecolor='white', markeredgewidth=2, label='Waste Dump')
    ax2.plot(plant_x, plant_y, 'ko', markersize=15,
            markeredgecolor='white', markeredgewidth=2, label='Processing Plant')
    
    apply_minimalist_style_manual(ax2)
    ax2.set_xlabel('Easting (km)', fontsize=10)
    ax2.set_ylabel('Northing (km)', fontsize=10)
    ax2.set_title('Z-Score Anomaly Detection', 
                  fontsize=12, fontweight='bold', loc='center', pad=15)
    ax2.legend(loc='upper right', frameon=False, fontsize=8)
    ax2.set_aspect('equal')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Z-Score (σ)', fontsize=10)
    cbar2.outline.set_visible(False)
    
    plt.suptitle('Spatial Thermal Anomaly Analysis', 
                fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/16_thermal_anomaly_spatial.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Spatial thermal heatmap visualization saved")

def main():
    """Generate all visualizations for Blog 16."""
    set_tufte_defaults()
    print("="*70)
    print("Blog 16: Mine Thermal Anomaly (MODIS) - Visualizations")
    print("="*70)
    print()
    
    create_main_thermal_time_series()
    create_spatial_thermal_heatmap()
    
    print()
    print("="*70)
    print("All visualizations generated successfully!")
    print("="*70)
    print()
    print("Files created:")
    print("  - 16_thermal_anomaly_modis_main.png")
    print("  - 16_thermal_anomaly_spatial.png")

if __name__ == "__main__":
    main()

