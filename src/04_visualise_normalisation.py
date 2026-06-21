#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 09:14:48 2025

@author: vanessa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Normalization Process Visualization

This standalone script visualizes the vector normalization process step by step
for MCDM (Multi-Criteria Decision Making) data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def load_data(file_path):
    """
    Load data from the CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    DataFrame
        Loaded data
    """
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")
        
        # Ensure we have a scenario column
        if 'Scenario' in df.columns:
            df.rename(columns={'Scenario': 'scenario'}, inplace=True)
        
        # Ensure all indicator columns have _indicator suffix
        indicator_cols = [col for col in df.columns if col != 'scenario']
        new_cols = {}
        for col in indicator_cols:
            if not col.endswith('_indicator'):
                new_cols[col] = col + '_indicator'
        
        if new_cols:
            df.rename(columns=new_cols, inplace=True)
        
        # Select only NDC-LTP and VRE-65% and higher
        selected_scenarios = []
        for s in df['scenario'].unique():
            # Include NDC-LTP
            if 'NDC-LTP' in s:
                selected_scenarios.append(s)
            
            # Include VRE-65% and higher
            if 'VRE' in s:
                try:
                    percentage = s.split('-')[1].replace('%', '')
                    if int(percentage) >= 65:
                        selected_scenarios.append(s)
                except:
                    pass
        
        filtered_df = df[df['scenario'].isin(selected_scenarios)]
        print(f"Selected {len(filtered_df)} rows with scenarios: {selected_scenarios}")
        
        return filtered_df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def visualize_vector_normalization_process(data, output_dir, num_indicators=3):
    """
    Visualize the vector normalization process step by step.
    
    Parameters:
    -----------
    data : DataFrame
        Original data
    output_dir : str
        Directory to save visualizations
    num_indicators : int
        Number of indicators to visualize (default: 3)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get indicator columns
    indicator_cols = [col for col in data.columns if col.endswith('_indicator')]
    
    # Select a subset of indicators to visualize
    selected_indicators = indicator_cols[:min(num_indicators, len(indicator_cols))]
    
    # Identify cost criteria (for which lower values are better)
    cost_keywords = ['cost', 'emission', 'toxicity', 'pollution', 'risk', 'lole', 'eutrophication', 'ionising', 
                    'land_occupation', 'depletion']
    
    cost_criteria = []
    benefit_criteria = []
    
    for col in selected_indicators:
        col_name = col.replace('_indicator', '').lower()
        if any(keyword in col_name for keyword in cost_keywords):
            cost_criteria.append(col)
        else:
            benefit_criteria.append(col)
    
    # For each selected indicator, visualize the normalization process
    for col in selected_indicators:
        indicator_name = col.replace('_indicator', '')
        is_cost = col in cost_criteria
        
        # Step 1: Original data distribution
        original_values = data[col].values
        
        # Step 2: Calculate the square root of the sum of squares
        sqrt_sum_squares = np.sqrt(np.sum(original_values ** 2))
        
        # Step 3: Normalized values
        normalized_values = original_values / sqrt_sum_squares
        
        # Step 4: For cost criteria, convert to 1 - normalized value
        if is_cost:
            final_values = 1 - normalized_values
        else:
            final_values = normalized_values
        
        # Create a figure to visualize the process
        fig = plt.figure(figsize=(18, 10))
        
        # Define a custom colormap (blue to red gradient)
        colors = [(0.1, 0.1, 0.9), (0.9, 0.1, 0.1)]  # Blue to Red
        cmap = LinearSegmentedColormap.from_list("BlueToRed", colors, N=100)
        
        # Step 1: Original data
        plt.subplot(2, 2, 1)
        sns.histplot(original_values, kde=True, color='skyblue')
        plt.title(f'Step 1: Original Data\n{indicator_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        
        # Step 2: Square and sum visualization
        plt.subplot(2, 2, 2)
        squared_values = original_values ** 2
        plt.bar(range(len(squared_values)), squared_values, color='lightgreen', alpha=0.7)
        plt.axhline(y=np.sum(squared_values), color='red', linestyle='-', label=f'Sum of Squares: {np.sum(squared_values):.2e}')
        plt.axhline(y=sqrt_sum_squares**2, color='blue', linestyle='--', label=f'Square of √∑x²: {sqrt_sum_squares**2:.2e}')
        plt.title(f'Step 2: Square Values and Sum\n√∑x² = {sqrt_sum_squares:.2e}', fontsize=16, fontweight='bold')
        plt.xlabel('Data Points', fontsize=12)
        plt.ylabel('Squared Values', fontsize=12)
        plt.yscale('log')  # Log scale for better visualization
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        # Step 3: Divide by the square root of the sum of squares
        plt.subplot(2, 2, 3)
        plt.scatter(range(len(original_values)), original_values, color='blue', alpha=0.7, label='Original')
        plt.scatter(range(len(normalized_values)), normalized_values, color='green', alpha=0.7, label='Normalized')
        plt.title('Step 3: Divide by √∑x²\nNormalized = Original / √∑x²', fontsize=16, fontweight='bold')
        plt.xlabel('Data Points', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        # Step 4: Final values (with cost conversion if applicable)
        plt.subplot(2, 2, 4)
        
        # Sort the values for better visualization
        sort_idx = np.argsort(original_values)
        sorted_original = original_values[sort_idx]
        sorted_final = final_values[sort_idx]
        
        # Create a gradient color based on change magnitude
        color_values = np.abs(sorted_final - sorted_original) / max(np.abs(sorted_final - sorted_original).max(), 0.001)
        
        plt.scatter(range(len(sorted_original)), sorted_original, color='blue', alpha=0.7, label='Original', s=50)
        
        # Plot final values with color gradient
        for i in range(len(sorted_final)):
            plt.scatter(i, sorted_final[i], color=cmap(color_values[i]), alpha=0.7, s=50)
        
        # Connect original and final values with lines
        for i in range(len(sorted_original)):
            plt.plot([i, i], [sorted_original[i], sorted_final[i]], color=cmap(color_values[i]), alpha=0.4, linestyle='-')
        
        # Create a colorbar to show the magnitude of change
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Magnitude of Change', fontsize=10)
        
        if is_cost:
            plt.title(f'Step 4: Cost Criterion Conversion\nFinal = 1 - Normalized', fontsize=16, fontweight='bold')
        else:
            plt.title(f'Step 4: Benefit Criterion\nFinal = Normalized', fontsize=16, fontweight='bold')
        
        plt.xlabel('Data Points (sorted by original value)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        # Overall title
        plt.suptitle(f'Vector Normalization Process for {indicator_name}\n{"Cost Criterion" if is_cost else "Benefit Criterion"}', 
                    fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'vector_normalization_process_{indicator_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Vector normalization process visualizations saved in {output_dir}")


def create_vector_comparison_table(data, output_dir):
    """
    Create a visual table comparing original and normalized values.
    
    Parameters:
    -----------
    data : DataFrame
        Original data
    output_dir : str
        Directory to save visualizations
    """
    # Get indicator columns
    indicator_cols = [col for col in data.columns if col.endswith('_indicator')]
    
    # Select a subset of data for demonstration (first few rows)
    subset_data = data.iloc[:min(5, len(data))].copy()
    
    # Identify cost criteria
    cost_keywords = ['cost', 'emission', 'toxicity', 'pollution', 'risk', 'lole', 'eutrophication', 'ionising', 
                    'land_occupation', 'depletion']
    
    cost_criteria = []
    benefit_criteria = []
    
    for col in indicator_cols:
        col_name = col.replace('_indicator', '').lower()
        if any(keyword in col_name for keyword in cost_keywords):
            cost_criteria.append(col)
        else:
            benefit_criteria.append(col)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    
    # Create header row
    header = ['Scenario']
    for col in indicator_cols[:3]:  # Limit to first 3 indicators for clarity
        indicator_name = col.replace('_indicator', '')
        is_cost = col in cost_criteria
        header.append(f'{indicator_name}\nOriginal')
        header.append(f'{indicator_name}\nNormalized')
        header.append(f'{indicator_name}\nFinal {"(1-Norm)" if is_cost else ""}')
    
    # Add data rows
    for idx, row in subset_data.iterrows():
        row_data = [row['scenario']]
        
        for col in indicator_cols[:3]:  # Limit to first 3 indicators
            is_cost = col in cost_criteria
            
            # Original value
            original_value = row[col]
            row_data.append(f'{original_value:.3e}')
            
            # Calculate normalized value
            sqrt_sum_squares = np.sqrt(np.sum(data[col] ** 2))
            normalized_value = original_value / sqrt_sum_squares
            row_data.append(f'{normalized_value:.3e}')
            
            # Final value (with cost conversion if applicable)
            if is_cost:
                final_value = 1 - normalized_value
            else:
                final_value = normalized_value
            row_data.append(f'{final_value:.3e}')
        
        table_data.append(row_data)
    
    # Create the table
    table = ax.table(cellText=table_data, 
                    colLabels=header, 
                    loc='center',
                    cellLoc='center',
                    colColours=['#f2f2f2'] * len(header),
                    cellColours=[['#ffffff'] * len(header) for _ in range(len(table_data))])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Title
    plt.suptitle('Vector Normalization Process: Step-by-Step Example', fontsize=20, fontweight='bold')
    
    plt.savefig(os.path.join(output_dir, 'vector_normalization_table.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def create_animated_transformation(data, output_dir):
    """
    Create a visualization showing the transformation of values through normalization.
    
    Parameters:
    -----------
    data : DataFrame
        Original data
    output_dir : str
        Directory to save visualizations
    """
    # Get indicator columns
    indicator_cols = [col for col in data.columns if col.endswith('_indicator')]
    
    # Select one indicator for demonstration
    selected_indicator = indicator_cols[0]
    indicator_name = selected_indicator.replace('_indicator', '')
    
    # Identify if it's a cost criterion
    cost_keywords = ['cost', 'emission', 'toxicity', 'pollution', 'risk', 'lole', 'eutrophication', 'ionising', 
                    'land_occupation', 'depletion']
    
    is_cost = any(keyword in indicator_name.lower() for keyword in cost_keywords)
    
    # Get original values
    original_values = data[selected_indicator].values
    
    # Calculate the square root of the sum of squares
    sqrt_sum_squares = np.sqrt(np.sum(original_values ** 2))
    
    # Calculate normalized values
    normalized_values = original_values / sqrt_sum_squares
    
    # Calculate final values (with cost conversion if applicable)
    if is_cost:
        final_values = 1 - normalized_values
    else:
        final_values = normalized_values
    
    # Create a figure
    plt.figure(figsize=(16, 10))
    
    # Sort values for better visualization
    sort_idx = np.argsort(original_values)
    sorted_original = original_values[sort_idx]
    sorted_normalized = normalized_values[sort_idx]
    sorted_final = final_values[sort_idx]
    
    # Define x positions for the three stages
    x_orig = np.arange(len(sorted_original))
    x_norm = x_orig + len(sorted_original) + 10
    x_final = x_norm + len(sorted_normalized) + 10
    
    # Plot original values
    plt.scatter(x_orig, sorted_original, color='blue', s=80, label='Original Values')
    
    # Plot normalized values
    plt.scatter(x_norm, sorted_normalized, color='green', s=80, label='Normalized Values')
    
    # Plot final values
    if is_cost:
        plt.scatter(x_final, sorted_final, color='red', s=80, label='Final Values (1 - Normalized)')
    else:
        plt.scatter(x_final, sorted_final, color='red', s=80, label='Final Values (Same as Normalized)')
    
    # Connect the dots to show transformation
    for i in range(len(sorted_original)):
        # Connect original to normalized
        plt.plot([x_orig[i], x_norm[i]], [sorted_original[i], sorted_normalized[i]], 
                 'g--', alpha=0.4)
        
        # Connect normalized to final
        plt.plot([x_norm[i], x_final[i]], [sorted_normalized[i], sorted_final[i]], 
                 'r--', alpha=0.4)
    
    # Add labels
    plt.text(np.mean(x_orig), max(sorted_original) * 1.1, 'Original Data', 
             ha='center', fontsize=16, fontweight='bold')
    
    plt.text(np.mean(x_norm), max(sorted_normalized) * 1.1, 'Normalized: Original ÷ √∑x²', 
             ha='center', fontsize=16, fontweight='bold')
    
    if is_cost:
        plt.text(np.mean(x_final), max(sorted_final) * 1.1, 'Final: 1 - Normalized', 
                 ha='center', fontsize=16, fontweight='bold')
    else:
        plt.text(np.mean(x_final), max(sorted_final) * 1.1, 'Final: Same as Normalized', 
                 ha='center', fontsize=16, fontweight='bold')
    
    # Add arrows
    plt.annotate('', xy=(x_orig[-1] + 5, np.mean(sorted_original)), 
                 xytext=(x_norm[0] - 5, np.mean(sorted_normalized)),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=10))
    
    plt.annotate('', xy=(x_norm[-1] + 5, np.mean(sorted_normalized)), 
                 xytext=(x_final[0] - 5, np.mean(sorted_final)),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=10))
    
    # Add formulas
    plt.text((x_orig[-1] + x_norm[0]) / 2, min(sorted_original) * 0.8, 
             f'√∑x² = {sqrt_sum_squares:.3e}', 
             ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add title and labels
    plt.title(f'Vector Normalization Transformation for {indicator_name}\n{"Cost Criterion" if is_cost else "Benefit Criterion"}', 
              fontsize=20, fontweight='bold')
    plt.xlabel('Data Points (sorted by original value)', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    
    # Remove x-ticks for cleaner look
    plt.xticks([])
    
    # Add a grid
    plt.grid(alpha=0.3)
    
    # Add a legend
    plt.legend(fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'vector_normalization_transformation_{indicator_name}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def main(csv_path, output_dir='./Vector_Normalization_Visualization'):
    """
    Main function to visualize the vector normalization process.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    data = load_data(csv_path)
    
    # Visualize vector normalization process for selected indicators
    print("\nCreating vector normalization process visualizations...")
    visualize_vector_normalization_process(data, output_dir)
    
    # Create a table comparing original and normalized values
    print("\nCreating vector normalization comparison table...")
    create_vector_comparison_table(data, output_dir)
    
    # Create an animated transformation visualization
    print("\nCreating animated transformation visualization...")
    create_animated_transformation(data, output_dir)
    
    print(f"\nAll visualizations saved in {output_dir}")


if __name__ == "__main__":
    # Set CSV file path and output directory
    csv_path = '/Users/vanessa/Library/CloudStorage/OneDrive-UniversityCollegeLondon/research/data/paper3/MCDM_FINAL/combined_data.csv'  # Update with your file path
    output_dir = './Vector_Normalization_Visualization'
    
    # Run the visualization
    main(csv_path, output_dir)