"""
Model training module for house price prediction
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class HousePricePredictor:
    """
    House Price Predictor class supporting multiple regression algorithms
    """
    
    def __init__(self, model_type='linear'):
        """
        Initialize the predictor with specified model type
        
        Args:
            model_type (str): Type of model to use
                             Options: 'linear', 'ridge', 'lasso', 'decision_tree',
                                      'random_forest', 'gradient_boosting', 'svr'
        """
        self.model_type = model_type
        self.model = self._get_model(model_type)
        self.is_trained = False
        
    def _get_model(self, model_type):
        """
        Get the model instance based on model type
        
        Args:
            model_type (str): Type of model
            
        Returns:
            sklearn model instance
        """
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf')
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return models[model_type]
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (numpy.ndarray): Features to predict on
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
            
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test targets
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
            
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        return metrics


def train_multiple_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models
    
    Args:
        X_train (numpy.ndarray): Training features
        X_test (numpy.ndarray): Test features
        y_train (numpy.ndarray): Training targets
        y_test (numpy.ndarray): Test targets
        
    Returns:
        dict: Dictionary containing results for each model
    """
    model_types = ['linear', 'ridge', 'lasso', 'decision_tree', 
                   'random_forest', 'gradient_boosting', 'svr']
    
    results = {}
    
    for model_type in model_types:
        print(f"Training {model_type} model...")
        predictor = HousePricePredictor(model_type=model_type)
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        
        results[model_type] = {
            'model': predictor,
            'metrics': metrics
        }
        
        print(f"{model_type} - RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2_score']:.4f}")
    
    return results
