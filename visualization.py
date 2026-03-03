"""
Visualization module for model results and data analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_data_distribution(df, save_path='data_distribution.png'):
    """
    Plot the distribution of house prices
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['PRICE'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Price ($1000s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of House Prices in Boston Dataset')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Data distribution plot saved to {save_path}")


def plot_correlation_matrix(df, save_path='correlation_matrix.png'):
    """
    Plot correlation matrix of features
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix plot saved to {save_path}")


def plot_predictions(y_true, y_pred, model_name='Model', save_path='predictions.png'):
    """
    Plot predicted vs actual values
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($1000s)')
    plt.ylabel('Predicted Price ($1000s)')
    plt.title(f'{model_name}: Predicted vs Actual House Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Predictions plot saved to {save_path}")


def plot_residuals(y_true, y_pred, model_name='Model', save_path='residuals.png'):
    """
    Plot residuals (prediction errors)
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price ($1000s)')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals plot saved to {save_path}")


def plot_model_comparison(results, save_path='model_comparison.png'):
    """
    Plot comparison of different models
    
    Args:
        results (dict): Dictionary containing results for each model
        save_path (str): Path to save the plot
    """
    model_names = list(results.keys())
    rmse_values = [results[model]['metrics']['rmse'] for model in model_names]
    r2_values = [results[model]['metrics']['r2_score'] for model in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE comparison
    ax1.bar(model_names, rmse_values, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Model Comparison: Root Mean Squared Error')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # R2 comparison
    ax2.bar(model_names, r2_values, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Model Comparison: R² Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to {save_path}")


def plot_feature_importance(model, feature_names, save_path='feature_importance.png'):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): Names of features
        save_path (str): Path to save the plot
    """
    if not hasattr(model.model, 'feature_importances_'):
        print("This model doesn't support feature importance visualization")
        return
        
    importances = model.model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[indices], color='steelblue', edgecolor='black')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")
