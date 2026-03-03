# House Price Prediction using Boston Housing Dataset

A comprehensive machine learning project for predicting house prices using the Boston Housing dataset. This project implements multiple regression algorithms, compares their performance, and provides detailed visualizations.

## Features

- **Multiple ML Models**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, and SVR
- **Data Preprocessing**: Automatic data loading, splitting, and feature scaling
- **Model Evaluation**: Comprehensive metrics including RMSE, MAE, and R² score
- **Visualizations**: 
  - Data distribution plots
  - Correlation matrix
  - Prediction vs actual plots
  - Residual plots
  - Model comparison charts
  - Feature importance (for tree-based models)

## Dataset

The Boston Housing dataset contains information about houses in Boston suburbs:
- **Samples**: 506 instances
- **Features**: 13 numeric/categorical predictive features
- **Target**: Median house value (in $1000s)

### Features Description:
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
  - **Note**: This feature has been identified as ethically problematic. The original dataset creators assumed racial self-segregation positively impacted house prices, which is a racist assumption. This is one of the primary reasons the Boston Housing dataset was removed from scikit-learn in version 1.2. This implementation is provided for educational purposes to understand historical datasets and their ethical issues.
- **LSTAT**: Percentage of lower status of the population

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hari-yadav-tech/House_price_prediction.git
cd House_price_prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to execute the complete pipeline:

```bash
python main.py
```

This will:
1. Load the Boston Housing dataset
2. Display dataset information
3. Create data visualization plots
4. Preprocess the data (split and scale)
5. Train multiple models
6. Evaluate and compare models
7. Generate comprehensive visualizations
8. Save all plots to the `output/` directory

## Project Structure

```
House_price_prediction/
│
├── main.py              # Main script to run the pipeline
├── data_loader.py       # Data loading and preprocessing
├── model.py            # Model training and evaluation
├── visualization.py    # Visualization functions
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── .gitignore         # Git ignore file
└── output/            # Directory for generated plots (created automatically)
```

## Models Implemented

1. **Linear Regression**: Basic linear model
2. **Ridge Regression**: Linear regression with L2 regularization
3. **Lasso Regression**: Linear regression with L1 regularization
4. **Decision Tree**: Non-linear tree-based model
5. **Random Forest**: Ensemble of decision trees
6. **Gradient Boosting**: Sequential ensemble method
7. **Support Vector Regression (SVR)**: Kernel-based regression

## Results

The script automatically trains all models and compares their performance. Expected performance metrics:
- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R² Score**: Coefficient of determination (higher is better, max 1.0)

## Output

All visualizations are saved in the `output/` directory:
- `data_distribution.png`: Distribution of house prices
- `correlation_matrix.png`: Feature correlation heatmap
- `model_comparison.png`: Comparison of all models
- `{best_model}_predictions.png`: Predictions vs actual values
- `{best_model}_residuals.png`: Residual plot
- `{best_model}_feature_importance.png`: Feature importance (for tree-based models)

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## License

This project is open source and available for educational purposes.

## Author

Hari Yadav

## Acknowledgments

- Boston Housing dataset from scikit-learn
- Inspiration from various machine learning tutorials and courses
