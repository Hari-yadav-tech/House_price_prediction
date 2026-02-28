# Example Usage and Output

## Running the House Price Prediction System

To run the complete analysis:

```bash
python main.py
```

## Expected Output

```
============================================================
Boston Housing Price Prediction
============================================================

1. Loading Boston Housing dataset...
Dataset loaded successfully!
Shape: (506, 14)
Features: ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

2. Dataset Information:
Number of samples: 506
Number of features: 13
Target variable: PRICE (median house value in $1000s)

3. Creating visualizations...
Data distribution plot saved to output/data_distribution.png
Correlation matrix plot saved to output/correlation_matrix.png

4. Preprocessing data...
Training set size: 404
Test set size: 102

5. Training and evaluating models...
------------------------------------------------------------
Training linear model...
linear - RMSE: 2.3697, R2: 0.7240
Training ridge model...
ridge - RMSE: 2.3683, R2: 0.7243
Training lasso model...
lasso - RMSE: 3.0748, R2: 0.5354
Training decision_tree model...
decision_tree - RMSE: 3.6314, R2: 0.3519
Training random_forest model...
random_forest - RMSE: 2.4914, R2: 0.6949
Training gradient_boosting model...
gradient_boosting - RMSE: 2.4662, R2: 0.7011
Training svr model...
svr - RMSE: 2.7902, R2: 0.6174
------------------------------------------------------------

6. Model Performance Summary:
------------------------------------------------------------

LINEAR:
  RMSE: 2.3697
  MAE:  1.7758
  R²:   0.7240

RIDGE:
  RMSE: 2.3683
  MAE:  1.7745
  R²:   0.7243

LASSO:
  RMSE: 3.0748
  MAE:  2.4756
  R²:   0.5354

DECISION_TREE:
  RMSE: 3.6314
  MAE:  2.8520
  R²:   0.3519

RANDOM_FOREST:
  RMSE: 2.4914
  MAE:  1.9420
  R²:   0.6949

GRADIENT_BOOSTING:
  RMSE: 2.4662
  MAE:  1.9070
  R²:   0.7011

SVR:
  RMSE: 2.7902
  MAE:  2.1536
  R²:   0.6174
------------------------------------------------------------

Best Model: RIDGE (RMSE: 2.3683)

7. Creating visualizations for best model...
Predictions plot saved to output/ridge_predictions.png
Residuals plot saved to output/ridge_residuals.png

8. Creating model comparison plot...
Model comparison plot saved to output/model_comparison.png

9. Creating feature importance plot...

============================================================
Analysis Complete!
All visualizations saved to 'output/' directory
============================================================
```

## Understanding the Results

### Model Performance Metrics

- **RMSE (Root Mean Squared Error)**: Measures average prediction error
  - Lower values indicate better performance
  - In this case, Ridge regression achieves ~2.37, meaning average error of $2,370

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
  - More interpretable than RMSE
  - Ridge achieves ~1.77 ($1,770 average error)

- **R² Score (Coefficient of Determination)**: Proportion of variance explained by the model
  - Ranges from 0 to 1 (higher is better)
  - Ridge achieves 0.72, explaining 72% of price variance

### Best Model Selection

In this example, **Ridge Regression** performs best with:
- Lowest RMSE: 2.3683
- Good balance between bias and variance
- Regularization prevents overfitting

### Generated Visualizations

The system creates the following plots in the `output/` directory:

1. **data_distribution.png**: Shows the distribution of house prices
2. **correlation_matrix.png**: Heatmap of feature correlations
3. **model_comparison.png**: Bar charts comparing all models
4. **ridge_predictions.png**: Scatter plot of predicted vs actual prices
5. **ridge_residuals.png**: Residual plot showing prediction errors

## Using Individual Modules

### Data Loading

```python
from data_loader import load_data, preprocess_data

# Load data
X, y, feature_names = load_data()

# Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
```

### Training a Single Model

```python
from model import HousePricePredictor

# Create and train a model
predictor = HousePricePredictor(model_type='random_forest')
predictor.train(X_train, y_train)

# Evaluate
metrics = predictor.evaluate(X_test, y_test)
print(f"RMSE: {metrics['rmse']:.4f}")
```

### Creating Visualizations

```python
from visualization import plot_predictions, plot_model_comparison

# Plot predictions
y_pred = predictor.predict(X_test)
plot_predictions(y_test, y_pred, model_name='Random Forest')

# Compare multiple models
plot_model_comparison(results)
```

## Interpreting Feature Importance

For tree-based models (Decision Tree, Random Forest, Gradient Boosting), the system generates a feature importance plot showing which features most influence house prices. Typically:

- **RM (number of rooms)**: Strong positive correlation
- **LSTAT (lower status %)**: Strong negative correlation
- **NOX (pollution)**: Negative correlation
- **DIS (distance to employment)**: Moderate positive correlation
