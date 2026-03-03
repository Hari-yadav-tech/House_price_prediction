"""
Main script for Boston Housing Price Prediction
"""
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_data, create_dataframe, preprocess_data, get_data_info
from model import HousePricePredictor, train_multiple_models
from visualization import (plot_data_distribution, plot_correlation_matrix, 
                          plot_predictions, plot_residuals, plot_model_comparison,
                          plot_feature_importance)


def main():
    """
    Main function to run the house price prediction pipeline
    """
    print("=" * 60)
    print("Boston Housing Price Prediction")
    print("=" * 60)
    
    # Create output directory for plots
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Load data
    print("\n1. Loading Boston Housing dataset...")
    X, y, feature_names = load_data()
    df = create_dataframe(X, y, feature_names)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(feature_names)}")
    
    # 2. Display data information
    print("\n2. Dataset Information:")
    info = get_data_info(df)
    print(f"Number of samples: {info['shape'][0]}")
    print(f"Number of features: {info['shape'][1] - 1}")
    print(f"Target variable: PRICE (median house value in $1000s)")
    
    # 3. Visualize data
    print("\n3. Creating visualizations...")
    plot_data_distribution(df, save_path=os.path.join(output_dir, 'data_distribution.png'))
    plot_correlation_matrix(df, save_path=os.path.join(output_dir, 'correlation_matrix.png'))
    
    # 4. Preprocess data
    print("\n4. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # 5. Train and evaluate multiple models
    print("\n5. Training and evaluating models...")
    print("-" * 60)
    results = train_multiple_models(X_train, X_test, y_train, y_test)
    print("-" * 60)
    
    # 6. Find best model
    print("\n6. Model Performance Summary:")
    print("-" * 60)
    best_model_name = None
    best_rmse = float('inf')
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2_score']:.4f}")
        
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_model_name = model_name
    
    print("-" * 60)
    print(f"\nBest Model: {best_model_name.upper()} (RMSE: {best_rmse:.4f})")
    
    # 7. Visualize best model results
    print("\n7. Creating visualizations for best model...")
    best_model = results[best_model_name]['model']
    y_pred = best_model.predict(X_test)
    
    plot_predictions(y_test, y_pred, model_name=best_model_name.upper(),
                    save_path=os.path.join(output_dir, f'{best_model_name}_predictions.png'))
    plot_residuals(y_test, y_pred, model_name=best_model_name.upper(),
                  save_path=os.path.join(output_dir, f'{best_model_name}_residuals.png'))
    
    # 8. Plot model comparison
    print("\n8. Creating model comparison plot...")
    plot_model_comparison(results, save_path=os.path.join(output_dir, 'model_comparison.png'))
    
    # 9. Plot feature importance for tree-based models
    print("\n9. Creating feature importance plot...")
    if best_model_name in ['decision_tree', 'random_forest', 'gradient_boosting']:
        plot_feature_importance(best_model, feature_names,
                              save_path=os.path.join(output_dir, f'{best_model_name}_feature_importance.png'))
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"All visualizations saved to '{output_dir}/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
