#!/usr/bin/env python3
import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Generate visualizations for Thermal Anomaly Detection blog post.
Uses minimalist styling with serif fonts, clean axes, and high-quality output.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
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


def save_fig(filename):
    """Save plot in the standard minimalist format."""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def fetch_modis_lst_data(latitude, longitude, start_date, end_date):
    """Generate realistic MODIS LST data matching satellite characteristics."""
    dates = pd.date_range(start=start_date, end=end_date, freq='8D')
    base_temp_k = 295
    
    temperatures = []
    for date in dates:
        day_of_year = date.timetuple().tm_yday
        seasonal = 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        weather_noise = np.random.normal(0, 3)
        temp_k = base_temp_k + seasonal + weather_noise
        
        temperatures.append({
            'date': date,
            'lst_day_kelvin': temp_k,
            'lst_night_kelvin': temp_k - 12,
            'quality_flag': 0,
            'latitude': latitude,
            'longitude': longitude
        })
    
    return pd.DataFrame(temperatures)

def kelvin_to_celsius(kelvin):
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15

def calculate_thermal_baseline(thermal_data):
    """Calculate seasonal baseline statistics."""
    thermal_data['week_of_year'] = thermal_data['date'].dt.isocalendar().week
    
    seasonal_baseline = thermal_data.groupby('week_of_year').agg({
        'lst_day_celsius': ['mean', 'std'],
        'lst_night_celsius': ['mean', 'std']
    }).reset_index()
    
    seasonal_baseline.columns = ['week', 'day_mean', 'day_std', 'night_mean', 'night_std']
    
    overall_stats = {
        'day_mean': thermal_data['lst_day_celsius'].mean(),
        'day_std': thermal_data['lst_day_celsius'].std(),
        'day_p95': thermal_data['lst_day_celsius'].quantile(0.95),
        'day_p99': thermal_data['lst_day_celsius'].quantile(0.99),
        'night_mean': thermal_data['lst_night_celsius'].mean(),
        'night_std': thermal_data['lst_night_celsius'].std()
    }
    
    return {'seasonal': seasonal_baseline, 'overall': overall_stats}

def detect_thermal_anomalies(thermal_data, baseline, threshold_sigma=2.5):
    """Detect thermal anomalies using statistical thresholds."""
    result = thermal_data.copy()
    result['week_of_year'] = result['date'].dt.isocalendar().week
    
    result = result.merge(baseline['seasonal'], left_on='week_of_year', right_on='week', how='left')
    
    result['day_z_score'] = (result['lst_day_celsius'] - result['day_mean']) / result['day_std']
    result['night_z_score'] = (result['lst_night_celsius'] - result['night_mean']) / result['night_std']
    
    result['day_anomaly'] = result['day_z_score'] > threshold_sigma
    result['night_anomaly'] = result['night_z_score'] > threshold_sigma
    result['any_anomaly'] = result['day_anomaly'] | result['night_anomaly']
    result['anomaly_score'] = np.clip(result['day_z_score'] * 20, 0, 100)
    
    return result

def create_main_visualization():
    """Create main thermal anomaly detection visualization."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Fetch data
    thermal_data = fetch_modis_lst_data(-30.5, 121.5, '2022-01-01', '2024-01-01')
    thermal_data['lst_day_celsius'] = kelvin_to_celsius(thermal_data['lst_day_kelvin'])
    thermal_data['lst_night_celsius'] = kelvin_to_celsius(thermal_data['lst_night_kelvin'])
    
    # Calculate baseline
    baseline = calculate_thermal_baseline(thermal_data)
    
    # Inject anomaly in last 90 days
    thermal_with_anomaly = thermal_data.copy()
    recent_dates = thermal_with_anomaly['date'] > (thermal_with_anomaly['date'].max() - timedelta(days=90))
    thermal_with_anomaly.loc[recent_dates, 'lst_day_celsius'] += 8
    
    # Detect anomalies
    anomalies = detect_thermal_anomalies(thermal_with_anomaly, baseline)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top panel: Temperature time series with baseline
    ax1.plot(thermal_data['date'], thermal_data['lst_day_celsius'], 
             color='black', linewidth=1, label='Historical Temperature')
    ax1.axhline(y=baseline['overall']['day_mean'], color='gray', 
                linestyle='--', linewidth=0.8, label='Baseline Mean')
    ax1.axhline(y=baseline['overall']['day_p95'], color='gray', 
                linestyle=':', linewidth=0.8, label='95th Percentile')
    
    # Highlight anomaly period
    anomaly_period = anomalies[anomalies['any_anomaly']]
    ax1.scatter(anomaly_period['date'], anomaly_period['lst_day_celsius'], 
                color='black', s=50, marker='o', facecolors='white', 
                edgecolors='black', linewidths=1.5, label='Detected Anomalies', zorder=5)
    
    # Apply minimalist style manually to avoid duplicate titles
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_position(("outward", 5))
    ax1.spines["bottom"].set_position(("outward", 5))
    ax1.set_title('Thermal Anomaly Detection at Mine Tailings Dam', 
                  fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Land Surface Temperature (°C)', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    
    # Bottom panel: Anomaly score over time
    ax2.fill_between(anomalies['date'], 0, anomalies['anomaly_score'], 
                     color='gray', alpha=0.3)
    ax2.plot(anomalies['date'], anomalies['anomaly_score'], 
             color='black', linewidth=1)
    ax2.axhline(y=40, color='gray', linestyle='--', linewidth=0.8, label='Medium Risk')
    ax2.axhline(y=60, color='gray', linestyle='-.', linewidth=0.8, label='High Risk')
    
    # Apply minimalist style manually to avoid duplicate titles
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_position(("outward", 5))
    ax2.spines["bottom"].set_position(("outward", 5))
    ax2.set_title('Anomaly Severity Score', fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Anomaly Score (0-100)', fontsize=10)
    ax2.legend(loc='upper left', frameon=False, fontsize=9)
    ax2.set_ylim(0, 105)
    
    # Save
    save_fig('06_thermal_anomaly_main.png')
    print("✓ Created: 06_thermal_anomaly_main.png")

def create_trend_visualization():
    """Create thermal trend analysis visualization."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Fetch data with injected warming trend
    thermal_data = fetch_modis_lst_data(-30.5, 121.5, '2022-01-01', '2024-01-01')
    thermal_data['lst_day_celsius'] = kelvin_to_celsius(thermal_data['lst_day_kelvin'])
    thermal_data['lst_night_celsius'] = kelvin_to_celsius(thermal_data['lst_night_kelvin'])
    
    # Inject warming trend in recent data
    recent_mask = thermal_data['date'] > '2023-06-01'
    days_recent = (thermal_data.loc[recent_mask, 'date'] - thermal_data.loc[recent_mask, 'date'].min()).dt.days
    thermal_data.loc[recent_mask, 'lst_day_celsius'] += (days_recent / 180) * 6
    
    # Calculate rolling mean
    thermal_sorted = thermal_data.sort_values('date').copy()
    thermal_sorted['rolling_mean'] = thermal_sorted['lst_day_celsius'].rolling(window=6, min_periods=3).mean()
    
    # Calculate baseline
    baseline_data = thermal_data[thermal_data['date'] < '2023-06-01']
    baseline_mean = baseline_data['lst_day_celsius'].mean()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top panel: Temperature with trend
    ax1.plot(thermal_sorted['date'], thermal_sorted['lst_day_celsius'], 
             color='lightgray', linewidth=0.8, label='Raw Temperature')
    ax1.plot(thermal_sorted['date'], thermal_sorted['rolling_mean'], 
             color='black', linewidth=1.5, label='6-Period Moving Average')
    ax1.axhline(y=baseline_mean, color='gray', linestyle='--', 
                linewidth=0.8, label='Historical Baseline')
    
    # Highlight warming period
    warming_period = thermal_sorted[thermal_sorted['date'] > '2023-06-01']
    ax1.axvspan(warming_period['date'].min(), warming_period['date'].max(), 
                alpha=0.1, color='gray', label='Warming Period')
    
    # Apply minimalist style manually to avoid duplicate titles
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_position(("outward", 5))
    ax1.spines["bottom"].set_position(("outward", 5))
    ax1.set_title('Thermal Trend Analysis: Tailings Dam', 
                  fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Land Surface Temperature (°C)', fontsize=10)
    ax1.legend(loc='upper left', frameon=False, fontsize=9)
    
    # Bottom panel: Deviation from baseline
    thermal_sorted['deviation'] = thermal_sorted['lst_day_celsius'] - baseline_mean
    
    # Bar chart showing positive and negative deviations
    colors = ['black' if x >= 0 else 'gray' for x in thermal_sorted['deviation']]
    ax2.bar(thermal_sorted['date'], thermal_sorted['deviation'], 
            color=colors, width=6, alpha=0.6)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    
    # Apply minimalist style manually to avoid duplicate titles
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_position(("outward", 5))
    ax2.spines["bottom"].set_position(("outward", 5))
    ax2.set_title('Temperature Deviation from Baseline', 
                  fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Temperature Deviation (°C)', fontsize=10)
    
    # Save
    save_fig('06_thermal_anomaly_accuracy.png')
    print("✓ Created: 06_thermal_anomaly_accuracy.png")

def main():
    """Generate all visualizations."""
    set_tufte_defaults()
    print("=" * 60)
    print("THERMAL ANOMALY DETECTION - VISUALIZATION GENERATION")
    print("=" * 60)
    print()
    
    # Set serif font globally
    plt.rcParams['font.family'] = 'serif'
    
    print("Creating visualizations...")
    create_main_visualization()
    create_trend_visualization()
    
    print()
    print("=" * 60)
    print("All visualizations created successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

