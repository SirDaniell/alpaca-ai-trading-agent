"""
ML Timeframe Analyzer - Production Implementation

Uses industry-standard ML libraries:
- scikit-learn for k-NN classification
- statsmodels for ARIMA forecasting
- pandas-ta for technical indicators
- scipy for statistical analysis
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats
from scipy.signal import find_peaks
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, List, Any, Optional
import logging

from app.core.processing.tasks import TaskStore

logger = logging.getLogger(__name__)


class LorentzianDistance:
    """Custom Lorentzian distance metric for scikit-learn"""
    
    def __call__(self, X, Y):
        """
        Lorentzian distance: sum(log(1 + |x_i - y_i|))
        More robust to outliers than Euclidean distance
        
        Note: scikit-learn's pairwise_distances passes 1D arrays for X and Y
        """
        # Handle both 1D and 2D arrays
        diff = np.abs(X - Y)
        if diff.ndim == 1:
            return np.sum(np.log(1 + diff))
        else:
            return np.sum(np.log(1 + diff), axis=1)


class TimeframeMLAnalyzer:
    """
    Comprehensive ML analysis for a single timeframe using production libraries
    
    Algorithms:
    - k-NN Classification (Euclidean/Lorentzian distance)
    - ARIMA Time Series Forecasting
    - Mean Reversion Analysis (Z-Score)
    """
    
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.symbol = config.get('symbol')
        self.timeframe = config.get('timeframe')
        self.context_window = config.get('context_window', 2000)
        self.current_window = config.get('current_window', 30)
        self.models = config.get('models', {})
        self.indicators = config.get('indicators', ['rsi', 'cci', 'adx', 'wavetrend'])
        self.knn_config = config.get('knn_config', {})
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def calculate_features(self) -> pd.DataFrame:
        """
        Calculate technical indicators using pandas-ta
        
        Returns:
            DataFrame with calculated indicators
        """
        logger.info(f"Calculating features for {len(self.df)} bars")
        
        # RSI - Relative Strength Index
        if 'rsi' in self.indicators:
            self.df['rsi'] = ta.rsi(self.df['close'], length=14)
        
        # CCI - Commodity Channel Index
        if 'cci' in self.indicators:
            self.df['cci'] = ta.cci(
                self.df['high'], 
                self.df['low'], 
                self.df['close'], 
                length=20
            )
        
        # ADX - Average Directional Index
        if 'adx' in self.indicators:
            adx_df = ta.adx(
                self.df['high'], 
                self.df['low'], 
                self.df['close'], 
                length=14
            )
            if adx_df is not None and 'ADX_14' in adx_df.columns:
                self.df['adx'] = adx_df['ADX_14']
        
        # WaveTrend - Custom oscillator
        if 'wavetrend' in self.indicators:
            hlc3 = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            esa = ta.ema(hlc3, length=10)
            d = ta.ema(np.abs(hlc3 - esa), length=10)
            ci = (hlc3 - esa) / (0.015 * d + 1e-10)  # Avoid division by zero
            self.df['wavetrend'] = ta.ema(ci, length=11)
        
        # Drop NaN values
        df_clean = self.df.dropna()
        logger.info(f"Features calculated. Clean data: {len(df_clean)} bars")
        
        return df_clean
    
    def analyze_knn(self, task_id: str, progress_store: TaskStore) -> Dict:
        """
        k-NN classification using scikit-learn with custom distance metrics
        
        Returns:
            Dict with prediction, confidence, neighbors, and feature importance
        """
        progress_store.update_task(task_id, progress=10, message="Extracting features for k-NN...")
        
        # Calculate features
        df_features = self.calculate_features()
        
        if len(df_features) < 100:
            logger.warning(f"Insufficient data for k-NN: {len(df_features)} bars")
            return {
                'prediction': 'neutral',
                'confidence': 0.0,
                'neighbors': [],
                'feature_importance': []
            }
        
        # Extract feature vectors
        feature_cols = [col for col in self.indicators if col in df_features.columns]
        if not feature_cols:
            raise ValueError(f"No valid features found. Available: {df_features.columns.tolist()}")
        
        features = df_features[feature_cols].values
        
        # Normalize features to [0, 1]
        scaler = MinMaxScaler()
        features_norm = scaler.fit_transform(features)
        
        progress_store.update_task(task_id, progress=20, message="Running k-NN classification...")
        
        # k-NN parameters
        k = self.knn_config.get('k', 8)
        metric = self.knn_config.get('metric', 'lorentzian')
        prediction_horizon = self.knn_config.get('prediction_horizon', 4)
        max_bars_back = min(500, len(features_norm) - prediction_horizon - 10)
        
        if max_bars_back < k:
            logger.warning(f"Insufficient historical data: {max_bars_back} < {k}")
            return {
                'prediction': 'neutral',
                'confidence': 0.0,
                'neighbors': [],
                'feature_importance': []
            }
        
        # Create labels based on future price movement with dynamic thresholding
        labels = []
        price_changes = df_features['close'].pct_change().dropna()
        threshold = price_changes.std() * 0.5  # Dynamic threshold
        
        for i in range(len(df_features) - prediction_horizon):
            future_price = df_features.iloc[i + prediction_horizon]['close']
            current_price = df_features.iloc[i]['close']
            pct_change = (future_price - current_price) / current_price
            
            if pct_change > threshold:
                labels.append('bullish')
            elif pct_change < -threshold:
                labels.append('bearish')
            else:
                labels.append('neutral')
        
        # Prepare training data
        X_train = features_norm[:-prediction_horizon]
        y_train = np.array(labels)
        X_test = features_norm[-1].reshape(1, -1)  # Latest bar
        
        # Train k-NN classifier
        if metric == 'lorentzian':
            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric=LorentzianDistance(),
                algorithm='brute'  # Required for custom metrics
            )
        else:  # euclidean
            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric='euclidean',
                algorithm='auto'
            )
        
        knn.fit(X_train, y_train)
        
        # Make prediction
        prediction = knn.predict(X_test)[0]
        probabilities = knn.predict_proba(X_test)[0]
        confidence = probabilities.max()
        
        # Get nearest neighbors
        distances, indices = knn.kneighbors(X_test, n_neighbors=k)
        neighbors = [
            {
                'distance': float(distances[0][i]),
                'label': y_train[indices[0][i]],
                'index': int(indices[0][i])
            }
            for i in range(k)
        ]
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            X_train, 
            y_train, 
            feature_cols
        )
        
        progress_store.update_task(task_id, progress=30, message="k-NN analysis complete")
        
        logger.info(f"k-NN prediction: {prediction} (confidence: {confidence:.2f})")
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'neighbors': neighbors[:3],  # Top 3 neighbors
            'feature_importance': feature_importance
        }
    
    def analyze_arima(self, task_id: str, progress_store: TaskStore) -> Dict:
        """
        ARIMA time series forecasting using statsmodels
        
        Returns:
            Dict with forecast, confidence intervals, and trend
        """
        progress_store.update_task(task_id, progress=50, message="Running ARIMA forecast...")
        
        try:
            # Use closing prices
            prices = self.df['close'].values
            
            # Fit ARIMA model (1,1,1) - simple configuration
            model = ARIMA(prices, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast next 5 periods
            forecast_result = fitted_model.forecast(steps=5)
            forecast_values = forecast_result if isinstance(forecast_result, np.ndarray) else [forecast_result]
            
            # Get confidence intervals
            forecast_obj = fitted_model.get_forecast(steps=5)
            conf_int = forecast_obj.conf_int()
            
            # Determine trend
            current_price = prices[-1]
            forecast_mean = np.mean(forecast_values)
            
            if forecast_mean > current_price * 1.001:
                trend = 'bullish'
            elif forecast_mean < current_price * 0.999:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Calculate volatility
            returns = pd.Series(prices).pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))  # Annualized
            
            progress_store.update_task(task_id, progress=60, message="ARIMA forecast complete")
            
            logger.info(f"ARIMA trend: {trend}, volatility: {volatility:.4f}")
            
            return {
                'forecast': [float(x) for x in forecast_values],
                'confidence_intervals': {
                    'lower': [float(x) for x in (conf_int.iloc[:, 0] if hasattr(conf_int, 'iloc') else conf_int[:, 0])],
                    'upper': [float(x) for x in (conf_int.iloc[:, 1] if hasattr(conf_int, 'iloc') else conf_int[:, 1])]
                },
                'trend': trend,
                'volatility': volatility,
                'accuracy': float(fitted_model.aic)  # AIC as accuracy proxy
            }
            
        except Exception as e:
            logger.error(f"ARIMA analysis failed: {e}")
            # Return neutral forecast on error
            return {
                'forecast': [float(self.df['close'].iloc[-1])] * 5,
                'confidence_intervals': {'lower': [0.0] * 5, 'upper': [0.0] * 5},
                'trend': 'neutral',
                'volatility': 0.0,
                'accuracy': 0.0
            }
    
    def analyze_mean_reversion(self, task_id: str, progress_store: TaskStore) -> Dict:
        """
        Mean reversion analysis using z-score
        
        Returns:
            Dict with signal, z-score, expected return, and statistics
        """
        progress_store.update_task(task_id, progress=70, message="Analyzing mean reversion...")
        
        # Calculate rolling statistics
        window = 20
        prices = self.df['close']
        
        mean = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        
        # Current z-score
        current_price = prices.iloc[-1]
        current_mean = mean.iloc[-1]
        current_std = std.iloc[-1]
        
        if current_std == 0 or np.isnan(current_std):
            z_score = 0.0
        else:
            z_score = (current_price - current_mean) / current_std
        
        # Determine signal
        if z_score > 2:
            signal = 'overbought'
        elif z_score < -2:
            signal = 'oversold'
        else:
            signal = 'neutral'
        
        # Expected return to mean
        expected_return = -z_score * current_std / current_price if current_price != 0 else 0.0
        
        # Estimate time to revert (simplified)
        time_to_revert = int(abs(z_score) * 2) if abs(z_score) > 1 else 1
        
        progress_store.update_task(task_id, progress=80, message="Mean reversion analysis complete")
        
        logger.info(f"Mean reversion signal: {signal}, z-score: {z_score:.2f}")
        
        return {
            'signal': signal,
            'z_score': float(z_score),
            'expected_return': float(expected_return),
            'time_to_revert': time_to_revert,
            'mean': float(current_mean),
            'std_dev': float(current_std)
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """
        Calculate comprehensive risk metrics
        
        Returns:
            Dict with risk score, volatility, drawdown, etc.
        """
        returns = self.df['close'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = float(returns.std() * np.sqrt(252))
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())
        
        # Risk score (0-1, higher = riskier)
        risk_score = min(1.0, (volatility * 10 + abs(max_drawdown)) / 2)
        
        return {
            'score': risk_score,
            'volatility': volatility,
            'drawdown_risk': max_drawdown,
            'timeframe_alignment': 0.7,  # Placeholder - would compare with other timeframes
            'uncertainty': 0.3  # Placeholder - would calculate from model disagreement
        }
    
    def calculate_consensus(self, results: Dict) -> Dict:
        """
        Calculate consensus from all model predictions
        
        Returns:
            Dict with overall direction, confidence, and agreement
        """
        predictions = []
        
        if 'knn' in results and results['knn']:
            predictions.append(results['knn']['prediction'])
        
        if 'arima' in results and results['arima']:
            predictions.append(results['arima']['trend'])
        
        if 'meanReversion' in results and results['meanReversion']:
            mr_signal = results['meanReversion']['signal']
            if mr_signal == 'overbought':
                predictions.append('bearish')
            elif mr_signal == 'oversold':
                predictions.append('bullish')
            else:
                predictions.append('neutral')
        
        if not predictions:
            return {
                'direction': 'neutral',
                'confidence': 0.0,
                'agreement': 0.0
            }
        
        # Count votes
        from collections import Counter
        votes = Counter(predictions)
        top_prediction, top_count = votes.most_common(1)[0]
        
        confidence = top_count / len(predictions)
        agreement = top_count / len(predictions)
        
        return {
            'direction': top_prediction,
            'confidence': float(confidence),
            'agreement': float(agreement)
        }
    
    def _calculate_feature_importance(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> List[Dict]:
        """
        Calculate feature importance using variance-based method
        
        Returns:
            List of dicts with feature names and weights
        """
        importance = []
        
        for i, name in enumerate(feature_names):
            # Calculate variance of feature
            variance = np.var(X[:, i])
            importance.append({
                'feature': name,
                'weight': float(variance)
            })
        
        # Normalize weights to sum to 1
        total = sum(item['weight'] for item in importance)
        if total > 0:
            for item in importance:
                item['weight'] /= total
        
        # Sort by weight descending
        importance.sort(key=lambda x: x['weight'], reverse=True)
        
        return importance
    
    def run_full_analysis(self, task_id: str, progress_store: TaskStore) -> Dict:
        """
        Run all enabled ML models and calculate consensus
        
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting full ML analysis for {self.symbol} {self.timeframe}")
        
        results = {
            'timeframe': self.timeframe,
            'symbol': self.symbol,
            'window_size': self.current_window,
            'timestamp': int(pd.Timestamp.now().timestamp())
        }
        
        try:
            # Run k-NN if enabled
            if self.models.get('knn', True):
                results['knn'] = self.analyze_knn(task_id, progress_store)
            
            # Run ARIMA if enabled
            if self.models.get('arima', True):
                results['arima'] = self.analyze_arima(task_id, progress_store)
            
            # Run Mean Reversion if enabled
            if self.models.get('meanReversion', True):
                results['meanReversion'] = self.analyze_mean_reversion(task_id, progress_store)
            
            # Calculate consensus
            progress_store.update_task(task_id, progress=90, message="Calculating consensus...")
            results['consensus'] = self.calculate_consensus(results)
            
            # Calculate risk metrics
            results['risk'] = self.calculate_risk_metrics()
            
            logger.info(f"Analysis complete. Consensus: {results['consensus']['direction']}")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise
        
        return results
