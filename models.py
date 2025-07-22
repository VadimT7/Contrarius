"""Machine learning models for Contrarius betting engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
from datetime import datetime
import logging

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import config
from database import db_manager, ModelPerformance

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all prediction models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.version = f"v1.0_{datetime.now().strftime('%Y%m%d')}"
        
    def preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Preprocess features for model."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def calibrate(self, X: np.ndarray, y: np.ndarray, method: str = "isotonic"):
        """Calibrate model predictions."""
        self.calibrated_model = CalibratedClassifierCV(
            self.model, 
            method=method, 
            cv=3
        )
        self.calibrated_model.fit(X, y)
    
    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """Get probability predictions."""
        if calibrated and self.calibrated_model:
            return self.calibrated_model.predict_proba(X)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance
    
    def save(self, path: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'version': self.version
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.calibrated_model = model_data.get('calibrated_model')
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.version = model_data['version']


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""
    
    def __init__(self):
        super().__init__("lightgbm")
        
    def train(self, X: pd.DataFrame, y: np.ndarray, params: Optional[Dict] = None):
        """Train LightGBM model."""
        X_processed = self.preprocess_features(X, fit=True)
        
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            default_params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.log_evaluation(0)],
        )
        
        # Calculate feature importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(X.columns, importance / importance.sum()))
        
        # Calibrate
        self.calibrate(X_processed, y, config.model.calibration_method)


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self):
        super().__init__("xgboost")
        
    def train(self, X: pd.DataFrame, y: np.ndarray, params: Optional[Dict] = None):
        """Train XGBoost model."""
        X_processed = self.preprocess_features(X, fit=True)
        
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': 42,
            'use_label_encoder': False
        }
        
        if params:
            default_params.update(params)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(X.columns, importance / importance.sum()))
        
        # Calibrate
        self.calibrate(X_processed, y, config.model.calibration_method)


class CatBoostModel(BaseModel):
    """CatBoost model implementation."""
    
    def __init__(self):
        super().__init__("catboost")
        
    def train(self, X: pd.DataFrame, y: np.ndarray, params: Optional[Dict] = None):
        """Train CatBoost model."""
        X_processed = self.preprocess_features(X, fit=True)
        
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'random_seed': 42,
            'early_stopping_rounds': 50,
            'verbose': False
        }
        
        if params:
            default_params.update(params)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = CatBoostClassifier(**default_params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(X.columns, importance / importance.sum()))
        
        # Calibrate
        self.calibrate(X_processed, y, config.model.calibration_method)


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def __init__(self):
        super().__init__("random_forest")
        
    def train(self, X: pd.DataFrame, y: np.ndarray, params: Optional[Dict] = None):
        """Train Random Forest model."""
        X_processed = self.preprocess_features(X, fit=True)
        
        default_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_processed, y)
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(X.columns, importance / importance.sum()))
        
        # Calibrate
        self.calibrate(X_processed, y, config.model.calibration_method)


class NeuralNetModel(BaseModel):
    """Neural Network model implementation."""
    
    def __init__(self):
        super().__init__("neural_net")
        
    def _build_model(self, input_dim: int) -> Sequential:
        """Build neural network architecture."""
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, X: pd.DataFrame, y: np.ndarray, params: Optional[Dict] = None):
        """Train neural network model."""
        X_processed = self.preprocess_features(X, fit=True)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = self._build_model(X_processed.shape[1])
        
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Simple feature importance based on weights
        weights = np.abs(self.model.layers[0].get_weights()[0])
        importance = np.mean(weights, axis=1)
        self.feature_importance = dict(zip(X.columns, importance / importance.sum()))
        
        # No direct calibration for Keras, wrap in sklearn
        from sklearn.base import BaseEstimator
        
        class KerasWrapper(BaseEstimator):
            def __init__(self, model):
                self.model = model
            
            def fit(self, X, y):
                return self
            
            def predict_proba(self, X):
                pred = self.model.predict(X, verbose=0)
                return np.hstack([1-pred, pred])
        
        self.sklearn_wrapper = KerasWrapper(self.model)
        self.calibrate(X_processed, y, config.model.calibration_method)
    
    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """Get probability predictions."""
        if calibrated and self.calibrated_model:
            return self.calibrated_model.predict_proba(X)
        
        pred = self.model.predict(X, verbose=0)
        return np.hstack([1-pred, pred])


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""
    
    def __init__(self):
        super().__init__("logistic_regression")
        
    def train(self, X: pd.DataFrame, y: np.ndarray, params: Optional[Dict] = None):
        """Train Logistic Regression model."""
        X_processed = self.preprocess_features(X, fit=True)
        
        default_params = {
            'penalty': 'l2',
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }
        
        if params:
            default_params.update(params)
        
        self.model = LogisticRegression(**default_params)
        self.model.fit(X_processed, y)
        
        # Feature importance from coefficients
        importance = np.abs(self.model.coef_[0])
        self.feature_importance = dict(zip(X.columns, importance / importance.sum()))
        
        # Calibrate
        self.calibrate(X_processed, y, config.model.calibration_method)


class ModelEnsemble:
    """Ensemble of multiple models with dynamic weighting."""
    
    def __init__(self):
        self.models = {
            'lightgbm': LightGBMModel(),
            'xgboost': XGBoostModel(),
            'catboost': CatBoostModel(),
            'random_forest': RandomForestModel(),
            'neural_net': NeuralNetModel(),
            'logistic_regression': LogisticRegressionModel()
        }
        self.weights = config.model.ensemble_weights.copy()
        self.performance_tracker = ModelPerformanceTracker()
        
    def train_all(self, X: pd.DataFrame, y: np.ndarray):
        """Train all models in ensemble."""
        logger.info("Training ensemble models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                model.train(X, y)
                logger.info(f"{name} training completed")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
    
    def predict_proba(self, X: pd.DataFrame, use_dynamic_weights: bool = True) -> Dict[str, Any]:
        """Get ensemble predictions with confidence scores."""
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                X_processed = model.preprocess_features(X)
                pred = model.predict_proba(X_processed, calibrated=True)
                predictions[name] = pred[:, 1]  # Probability of positive class
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
                predictions[name] = np.full(len(X), 0.5)
        
        # Update weights if using dynamic weighting
        if use_dynamic_weights:
            self._update_weights()
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += pred * self.weights[name]
        
        # Calculate prediction variance as uncertainty measure
        pred_array = np.array(list(predictions.values()))
        pred_variance = np.var(pred_array, axis=0)
        confidence = 1 - pred_variance  # Higher variance = lower confidence
        
        # Get feature importance
        feature_importance = self._aggregate_feature_importance()
        
        return {
            'probability': ensemble_pred,
            'confidence': confidence,
            'individual_predictions': predictions,
            'weights': self.weights.copy(),
            'feature_importance': feature_importance
        }
    
    def _update_weights(self):
        """Update model weights based on recent performance."""
        recent_performance = self.performance_tracker.get_recent_performance()
        
        if not recent_performance:
            return
        
        # Calculate performance scores
        scores = {}
        for name in self.models.keys():
            if name in recent_performance:
                # Combine accuracy and log loss for scoring
                accuracy = recent_performance[name].get('accuracy', 0.5)
                log_loss_score = recent_performance[name].get('log_loss', 1.0)
                
                # Lower log loss is better, so invert it
                score = accuracy * (1 / (log_loss_score + 0.01))
                scores[name] = score
            else:
                scores[name] = 1.0
        
        # Normalize scores to weights
        total_score = sum(scores.values())
        if total_score > 0:
            for name in self.models.keys():
                self.weights[name] = scores[name] / total_score
        
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def _aggregate_feature_importance(self) -> Dict[str, float]:
        """Aggregate feature importance across all models."""
        aggregated = {}
        
        for name, model in self.models.items():
            importance = model.get_feature_importance()
            weight = self.weights[name]
            
            for feature, score in importance.items():
                if feature not in aggregated:
                    aggregated[feature] = 0
                aggregated[feature] += score * weight
        
        # Normalize
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v/total for k, v in aggregated.items()}
        
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        predictions = self.predict_proba(X)
        y_pred = predictions['probability']
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred > 0.5),
            'auc_roc': roc_auc_score(y, y_pred),
            'log_loss': log_loss(y, y_pred),
            'brier_score': brier_score_loss(y, y_pred)
        }
        
        return metrics
    
    def save_models(self, directory: str):
        """Save all models to directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            model.save(os.path.join(directory, f"{name}_model.pkl"))
        
        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'version': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        joblib.dump(metadata, os.path.join(directory, 'ensemble_metadata.pkl'))
    
    def load_models(self, directory: str):
        """Load all models from directory."""
        import os
        
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}_model.pkl")
            if os.path.exists(path):
                model.load(path)
        
        # Load ensemble metadata
        metadata_path = os.path.join(directory, 'ensemble_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.weights = metadata['weights']


class ModelPerformanceTracker:
    """Track and analyze model performance over time."""
    
    def __init__(self):
        self.session = db_manager.get_session()
    
    def log_performance(self, model_name: str, sport: str, market_type: str, 
                       predictions: List[float], outcomes: List[int], bets: List[Dict]):
        """Log model performance metrics."""
        from utils import calculate_roi, calculate_calibration_error, calculate_brier_score
        
        # Calculate metrics
        accuracy = accuracy_score(outcomes, np.array(predictions) > 0.5)
        log_loss_score = log_loss(outcomes, predictions)
        brier_score = calculate_brier_score(predictions, outcomes)
        calibration_error = calculate_calibration_error(predictions, outcomes)
        
        # Calculate betting performance
        total_stake = sum(bet['stake'] for bet in bets)
        total_profit = sum(bet['profit'] for bet in bets if 'profit' in bet)
        roi = calculate_roi(total_profit, total_stake)
        
        # Store in database
        performance = ModelPerformance(
            model_name=model_name,
            date=datetime.utcnow(),
            sport=sport,
            market_type=market_type,
            total_predictions=len(predictions),
            correct_predictions=sum(1 for p, o in zip(predictions, outcomes) if (p > 0.5) == o),
            accuracy=accuracy,
            log_loss=log_loss_score,
            brier_score=brier_score,
            roi=roi,
            profit_loss=total_profit,
            calibration_error=calibration_error
        )
        
        self.session.add(performance)
        self.session.commit()
    
    def get_recent_performance(self, days: int = 7) -> Dict[str, Dict[str, float]]:
        """Get recent performance metrics for all models."""
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        performances = self.session.query(ModelPerformance).filter(
            ModelPerformance.date >= cutoff_date
        ).all()
        
        # Aggregate by model
        model_metrics = {}
        for perf in performances:
            if perf.model_name not in model_metrics:
                model_metrics[perf.model_name] = {
                    'accuracy': [],
                    'log_loss': [],
                    'roi': [],
                    'calibration_error': []
                }
            
            model_metrics[perf.model_name]['accuracy'].append(perf.accuracy)
            model_metrics[perf.model_name]['log_loss'].append(perf.log_loss)
            model_metrics[perf.model_name]['roi'].append(perf.roi)
            model_metrics[perf.model_name]['calibration_error'].append(perf.calibration_error)
        
        # Calculate averages
        for model, metrics in model_metrics.items():
            for metric, values in metrics.items():
                model_metrics[model][metric] = np.mean(values) if values else 0.0
        
        return model_metrics 