# src/models/trading_models.py
"""
Advanced machine learning models for crypto trading with proper backtesting.
Includes multiple model types and ensemble methods.
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# For advanced models (install separately if needed)
try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    warnings.warn("XGBoost/LightGBM not available. Install for better performance.")

logger = logging.getLogger(__name__)

class TradingModelPipeline:
    """
    Complete ML pipeline for crypto trading predictions.
    
    Features:
    1. Multiple model architectures
    2. Proper time series cross-validation
    3. Feature selection and engineering
    4. Model calibration
    5. Ensemble methods
    """
    
    def __init__(self, config, model_type: str = 'ensemble'):
        """
        Initialize trading model pipeline.
        
        Args:
            config: Trading configuration
            model_type: Type of model ('logistic', 'rf', 'gb', 'xgb', 'ensemble')
        """
        self.config = config
        self.model_type = model_type
        self.model = None
        self.feature_pipeline = None
        self.feature_names = None
        self.scaler = None
        self.metrics_history = []
        
    def build_model(self) -> Pipeline:
        """Build the specified model architecture."""
        
        # Feature preprocessing pipeline
        preprocessing = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
        ])
        
        # Select model based on type
        if self.model_type == 'logistic':
            base_model = LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            
        elif self.model_type == 'rf':
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
        elif self.model_type == 'gb':
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
            
        elif self.model_type == 'xgb' and ADVANCED_MODELS_AVAILABLE:
            base_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
        elif self.model_type == 'lgb' and ADVANCED_MODELS_AVAILABLE:
            base_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
            
        elif self.model_type == 'ensemble':
            # Create ensemble of models
            base_model = self._build_ensemble()
            
        else:
            # Default to decision tree
            base_model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42
            )
        
        # Calibrate probabilities for better probability estimates
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method='isotonic',
            cv=3
        )
        
        # Complete pipeline
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', calibrated_model)
        ])
        
        return pipeline
    
    def _build_ensemble(self):
        """Build an ensemble model combining multiple algorithms."""
        models = []
        
        # Logistic Regression
        models.append(('lr', LogisticRegression(
            C=0.1, class_weight='balanced', max_iter=1000
        )))
        
        # Random Forest
        models.append(('rf', RandomForestClassifier(
            n_estimators=50, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1
        )))
        
        # Gradient Boosting
        models.append(('gb', GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=5,
            random_state=42
        )))
        
        # Add XGBoost if available
        if ADVANCED_MODELS_AVAILABLE:
            models.append(('xgb', xgb.XGBClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=5,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )))
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',  # Use probability averaging
            n_jobs=-1
        )
        
        return ensemble
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
         feature_cols: Optional[List[str]] = None,
         timestamps: Optional[pd.Series] = None,
         cv_splits: int = 5) -> Dict[str, float]:
        """
        Train model with time series cross-validation including purging/embargo.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            feature_cols: List of feature columns to use
            timestamps: Timestamp series for purging/embargo (required)
            cv_splits: Number of CV splits
        """
        if timestamps is None:
            raise ValueError("timestamps required for proper time-series validation with purging/embargo")
        
        if feature_cols:
            X = X[feature_cols]
            self.feature_names = feature_cols
        else:
            X = X.select_dtypes(include=[np.number])
            self.feature_names = X.columns.tolist()
        
        logger.info(f"Training {self.model_type} model with {len(self.feature_names)} features")
        
        # Build model
        self.model = self.build_model()
        
        # Time series cross-validation with purging and embargo
        splits = self._purge_embargo_splits(X, y, timestamps, cv_splits, embargo_pct=0.01)
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train on fold
            self.model.fit(X_train, y_train)
            
            # Predict
            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val, y_pred, y_proba)
            metrics['fold'] = fold
            cv_metrics.append(metrics)
            
            logger.info(f"Fold {fold}: AUC={metrics['auc']:.3f}, Accuracy={metrics['accuracy']:.3f}")
        
        # Store metrics history
        self.metrics_history = cv_metrics
        
        # Calculate average metrics
        avg_metrics = {}
        for key in cv_metrics[0].keys():
            if key != 'fold':
                avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in cv_metrics])
                avg_metrics[f'std_{key}'] = np.std([m[key] for m in cv_metrics])
        
        # Final training on all data
        self.model.fit(X, y)
        
        return avg_metrics
    
    def _calculate_metrics(self,
                          y_true: pd.Series,
                          y_pred: np.ndarray,
                          y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'log_loss': log_loss(y_true, y_proba)
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Use same features as training
        if self.feature_names:
            X = X[self.feature_names]
        else:
            X = X.select_dtypes(include=[np.number])
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Extract the actual model from pipeline
        actual_model = self.model.named_steps['model']
        
        if hasattr(actual_model, 'estimator'):
            actual_model = actual_model.estimator
        elif hasattr(actual_model, 'base_estimator'):  # Backward compatibility
            actual_model = actual_model.base_estimator
        
        importance_scores = None
        
        # Get importance based on model type
        if hasattr(actual_model, 'feature_importances_'):
            importance_scores = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            importance_scores = np.abs(actual_model.coef_[0])
        elif self.model_type == 'ensemble' and hasattr(actual_model, 'estimators_'):
            # Average importance across ensemble members
            importances = []
            for name, estimator in actual_model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
                elif hasattr(estimator, 'coef_'):
                    importances.append(np.abs(estimator.coef_[0]))
            
            if importances:
                importance_scores = np.mean(importances, axis=0)
        
        if importance_scores is None:
            logger.warning("Cannot extract feature importance from this model type")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Normalize to sum to 1
        importance_df['importance_normalized'] = (
            importance_df['importance'] / importance_df['importance'].sum()
        )
        
        return importance_df
    
    def save_model(self, path: Optional[Path] = None):
        """Save trained model and metadata."""
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            path = self.config.models_dir / f"{self.model_type}_model_{datetime.now():%Y%m%d_%H%M%S}"
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, path / 'model.pkl')
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics_history': self.metrics_history,
            'config': {
                'enter_threshold': self.config.enter_threshold,
                'exit_threshold': self.config.exit_threshold,
                'symbols': self.config.symbols
            },
            'trained_at': datetime.now().isoformat()
        }
        
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save feature importance
        importance_df = self.get_feature_importance()
        if not importance_df.empty:
            importance_df.to_csv(path / 'feature_importance.csv', index=False)
        
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: Path):
        """Load saved model and metadata."""
        path = Path(path)
        
        # Load model
        self.model = joblib.load(path / 'model.pkl')
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.feature_names = metadata['feature_names']
        self.metrics_history = metadata.get('metrics_history', [])
        
        logger.info(f"Model loaded from {path}")

    def _purge_embargo_splits(self, X: pd.DataFrame, y: pd.Series, 
                         timestamps: pd.Series, n_splits: int = 5,
                         embargo_pct: float = 0.01) -> List[Tuple]:
        """
        Create time series splits with purging and embargo.
        
        Purging: Remove samples from training that have overlapping 
        forward returns with validation set.
        
        Embargo: Add gap after training set to prevent information leakage.
        
        Args:
            X: Features
            y: Labels  
            timestamps: Timestamp series (same index as X)
            n_splits: Number of CV splits
            embargo_pct: Percentage of samples to embargo (e.g., 0.01 = 1%)
        
        Returns:
            List of (train_idx, val_idx) tuples
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, val_idx in tscv.split(X):
            # Get timestamps
            train_times = timestamps.iloc[train_idx]
            val_times = timestamps.iloc[val_idx]
            
            # Embargo: remove last embargo_pct of training samples
            embargo_size = int(len(train_idx) * embargo_pct)
            if embargo_size > 0:
                train_idx = train_idx[:-embargo_size]
            
            # Purging: remove training samples that overlap with validation
            # For forward_return_6, we need to purge last 6 samples before val
            val_start = val_times.min()
            purge_threshold = val_start - pd.Timedelta(hours=6)  # Adjust based on max horizon
            
            # Keep only training samples before purge threshold
            train_mask = train_times < purge_threshold
            train_idx = train_idx[train_mask.values]
            
            splits.append((train_idx, val_idx))
        
        return splits

class TradingBacktester:
    """
    Sophisticated backtesting system for crypto trading strategies.
    
    Features:
    1. Realistic transaction costs and slippage
    2. Position sizing and risk management
    3. Multiple performance metrics
    4. Trade analysis and visualization
    """
    
    def __init__(self, config):
        """Initialize backtester with configuration."""
        self.config = config
        self.results = None
        
    def backtest(self,
                df: pd.DataFrame,
                probability_col: str = 'probability',
                strategy: str = 'threshold',
                **strategy_params) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with predictions and market data
            probability_col: Column with predicted probabilities
            strategy: Trading strategy type
            **strategy_params: Strategy-specific parameters
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest with {strategy} strategy")
        
        # Copy data to avoid modifications
        df = df.copy().sort_values(['symbol', 'timestamp'])
        
        # Generate trading signals
        if strategy == 'threshold':
            df = self._threshold_strategy(df, probability_col, **strategy_params)
        elif strategy == 'momentum':
            df = self._momentum_strategy(df, probability_col, **strategy_params)
        elif strategy == 'mean_reversion':
            df = self._mean_reversion_strategy(df, probability_col, **strategy_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Calculate returns
        df = self._calculate_returns(df)
        
        # Analyze results
        results = self._analyze_results(df)
        
        self.results = results
        return results
    
    def _threshold_strategy(self,
                          df: pd.DataFrame,
                          prob_col: str,
                          enter_threshold: Optional[float] = None,
                          exit_threshold: Optional[float] = None,
                          min_hold: int = 0) -> pd.DataFrame:
        """
        Simple threshold-based trading strategy.
        
        Enter long when probability > enter_threshold
        Exit when probability < exit_threshold
        """
        if enter_threshold is None:
            enter_threshold = self.config.enter_threshold
        if exit_threshold is None:
            exit_threshold = self.config.exit_threshold
        
        positions = []
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            probs = df.loc[mask, prob_col].values
            
            position = 0
            hold_count = 0
            symbol_positions = []
            
            for prob in probs:
                if position == 0:
                    # Not in position - check entry
                    if prob > enter_threshold:
                        position = 1
                        hold_count = 0
                else:
                    # In position - check exit
                    hold_count += 1
                    if hold_count >= min_hold and prob < exit_threshold:
                        position = 0
                        hold_count = 0
                
                symbol_positions.append(position)
            
            positions.extend(symbol_positions)
        
        df['position'] = positions
        
        # Detect trades (position changes)
        df['position_prev'] = df.groupby('symbol')['position'].shift(1).fillna(0)
        df['trade'] = (df['position'] != df['position_prev']).astype(int)
        
        return df
    
    def _momentum_strategy(self,
                         df: pd.DataFrame,
                         prob_col: str,
                         lookback: int = 3,
                         threshold: float = 0.6) -> pd.DataFrame:
        """
        Momentum-based strategy.
        
        Enter when probability is rising and above threshold.
        """
        df['prob_momentum'] = df.groupby('symbol')[prob_col].diff(lookback)
        df['position'] = ((df[prob_col] > threshold) & 
                         (df['prob_momentum'] > 0)).astype(int)
        
        df['position_prev'] = df.groupby('symbol')['position'].shift(1).fillna(0)
        df['trade'] = (df['position'] != df['position_prev']).astype(int)
        
        return df
    
    def _mean_reversion_strategy(self,
                                df: pd.DataFrame,
                                prob_col: str,
                                lookback: int = 20,
                                z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Mean reversion strategy.
        
        Trade when probability deviates significantly from recent average.
        """
        # Calculate z-score of probability
        rolling = df.groupby('symbol')[prob_col].rolling(lookback, min_periods=lookback//2)
        mean = rolling.mean().reset_index(level=0, drop=True)
        std = rolling.std().reset_index(level=0, drop=True)
        
        df['prob_zscore'] = (df[prob_col] - mean) / (std + 1e-9)
        
        # Enter when z-score is extreme
        df['position'] = (df['prob_zscore'] > z_threshold).astype(int)
        
        df['position_prev'] = df.groupby('symbol')['position'].shift(1).fillna(0)
        df['trade'] = (df['position'] != df['position_prev']).astype(int)
        
        return df
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns including costs."""
        # Base returns (already in dataset as forward_return_1)
        df['market_return'] = df.get('forward_return_1', 0)
        
        # Transaction costs
        fee = self.config.fee_per_side
        slippage = self.config.slippage_per_side
        total_cost = fee + slippage
        
        df['trade_cost'] = df['trade'] * total_cost
        
        # Strategy returns
        df['gross_return'] = df['position'] * df['market_return']
        df['net_return'] = df['gross_return'] - df['trade_cost']
        
        # Cumulative returns
        df['cumulative_market'] = df.groupby('symbol')['market_return'].cumsum()
        df['cumulative_gross'] = df.groupby('symbol')['gross_return'].cumsum()
        df['cumulative_net'] = df.groupby('symbol')['net_return'].cumsum()
        
        return df
    
    def _analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze backtest results and calculate metrics."""
        results = {}
        
        # Overall metrics
        results['overall'] = self._calculate_performance_metrics(df)
        
        # Per-symbol metrics
        results['by_symbol'] = {}
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            results['by_symbol'][symbol] = self._calculate_performance_metrics(symbol_df)
        
        # Trade analysis
        results['trades'] = self._analyze_trades(df)
        
        # Risk metrics
        results['risk'] = self._calculate_risk_metrics(df)
        
        # Save detailed results
        results['data'] = df
        
        return results
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for a strategy."""
        if df.empty:
            return {}
        
        # Basic metrics
        total_return = df['net_return'].sum()
        n_trades = df['trade'].sum()
        n_bars = len(df)
        
        # Win rate
        trade_returns = df[df['trade'] == 1]['gross_return']
        winning_trades = (trade_returns > 0).sum()
        total_trades = len(trade_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Sharpe ratio
        if df['net_return'].std() > 0:
            sharpe = np.sqrt(252) * df['net_return'].mean() / df['net_return'].std()
        else:
            sharpe = 0
        
        # Market exposure
        exposure = df['position'].mean()
        
        # Profit factor
        gross_profit = df[df['gross_return'] > 0]['gross_return'].sum()
        gross_loss = abs(df[df['gross_return'] < 0]['gross_return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return {
            'total_return': float(total_return),
            'n_trades': int(n_trades),
            'n_bars': int(n_bars),
            'win_rate': float(win_rate),
            'sharpe_ratio': float(sharpe),
            'exposure': float(exposure),
            'profit_factor': float(profit_factor),
            'avg_return_per_trade': float(total_return / n_trades) if n_trades > 0 else 0,
            'avg_return_per_bar': float(total_return / n_bars),
        }
    
    def _analyze_trades(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual trades."""
        trades = df[df['trade'] == 1].copy()
        
        if trades.empty:
            return {'n_trades': 0}
        
        # Find trade durations
        trade_durations = []
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol]
            positions = df[df['symbol'] == symbol]['position'].values
            
            for idx in symbol_trades.index:
                # Find how long position was held
                start_idx = df.index.get_loc(idx)
                duration = 1
                for i in range(start_idx + 1, len(positions)):
                    if positions[i] == 0:
                        break
                    duration += 1
                trade_durations.append(duration)
        
        return {
            'n_trades': len(trades),
            'trades_per_symbol': trades.groupby('symbol').size().to_dict(),
            'avg_duration': np.mean(trade_durations) if trade_durations else 0,
            'max_duration': np.max(trade_durations) if trade_durations else 0,
            'min_duration': np.min(trade_durations) if trade_durations else 0,
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics."""
        cumulative = df.groupby('symbol')['net_return'].cumsum()
        
        # Maximum drawdown
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (95% confidence)
        var_95 = df['net_return'].quantile(0.05)
        
        # Conditional Value at Risk
        cvar_95 = df[df['net_return'] <= var_95]['net_return'].mean()
        
        return {
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'return_volatility': float(df['net_return'].std()),
            'downside_deviation': float(df[df['net_return'] < 0]['net_return'].std()),
        }
    
    def generate_report(self) -> str:
        """Generate a text report of backtest results."""
        if self.results is None:
            return "No backtest results available"
        
        report = []
        report.append("=" * 60)
        report.append("BACKTEST RESULTS REPORT")
        report.append("=" * 60)
        
        # Overall performance
        overall = self.results['overall']
        report.append("\n### OVERALL PERFORMANCE ###")
        report.append(f"Total Return: {overall['total_return']:.2%}")
        report.append(f"Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
        report.append(f"Win Rate: {overall['win_rate']:.2%}")
        report.append(f"Number of Trades: {overall['n_trades']}")
        report.append(f"Market Exposure: {overall['exposure']:.2%}")
        report.append(f"Profit Factor: {overall['profit_factor']:.2f}")
        
        # Risk metrics
        risk = self.results['risk']
        report.append("\n### RISK METRICS ###")
        report.append(f"Max Drawdown: {risk['max_drawdown']:.2%}")
        report.append(f"VaR (95%): {risk['var_95']:.3%}")
        report.append(f"CVaR (95%): {risk['cvar_95']:.3%}")
        report.append(f"Return Volatility: {risk['return_volatility']:.3%}")
        
        # Per-symbol performance
        report.append("\n### PERFORMANCE BY SYMBOL ###")
        for symbol, metrics in self.results['by_symbol'].items():
            report.append(f"\n{symbol}:")
            report.append(f"  Return: {metrics['total_return']:.2%}")
            report.append(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            report.append(f"  Trades: {metrics['n_trades']}")
            report.append(f"  Win Rate: {metrics['win_rate']:.2%}")
        
        # Trade analysis
        trades = self.results['trades']
        report.append("\n### TRADE ANALYSIS ###")
        report.append(f"Total Trades: {trades['n_trades']}")
        report.append(f"Avg Trade Duration: {trades.get('avg_duration', 0):.1f} bars")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)