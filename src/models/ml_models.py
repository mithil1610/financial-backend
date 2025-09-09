import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from datetime import datetime, timedelta

class StockPredictor:
    """Base class for stock prediction models."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_names = []
        self.metrics = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training."""
        features = pd.DataFrame()
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['volatility'] = features['returns'].rolling(window=20).std()
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(df['close'])
        features['macd'], features['macd_signal'] = self.calculate_macd(df['close'])
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
        
        # Price ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Fundamental features (if available)
        if 'pe_ratio' in df.columns:
            features['pe_ratio'] = df['pe_ratio']
        if 'market_cap' in df.columns:
            features['market_cap_log'] = np.log(df['market_cap'])
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Remove NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        self.feature_names = features.columns.tolist()
        return features
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def train(self, df: pd.DataFrame):
        """Train the model - to be implemented by subclasses."""
        raise NotImplementedError
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> np.ndarray:
        """Make predictions - to be implemented by subclasses."""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat()
        }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.ticker = data['ticker']

class XGBoostPredictor(StockPredictor):
    """XGBoost-based stock predictor."""
    
    def __init__(self, ticker: str):
        super().__init__(ticker)
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.01,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
    
    def train(self, df: pd.DataFrame, target_days: int = 1):
        """Train XGBoost model."""
        # Prepare features
        features = self.prepare_features(df)
        
        # Create target (next day's return)
        target = df['close'].shift(-target_days) / df['close'] - 1
        
        # Remove NaN values
        mask = ~(target.isna() | features.isna().any(axis=1))
        X = features[mask]
        y = target[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_scaled, y,
            eval_set=[(X_scaled, y)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Calculate metrics
        predictions = self.model.predict(X_scaled)
        self.metrics = {
            'rmse': np.sqrt(np.mean((predictions - y) ** 2)),
            'mae': np.mean(np.abs(predictions - y)),
            'r2': 1 - (np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2))
        }
        
        return self.metrics
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> np.ndarray:
        """Make predictions for future days."""
        features = self.prepare_features(df)
        X_scaled = self.scaler.transform(features.iloc[-1:])
        
        predictions = []
        for _ in range(days):
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            # Simple approach: use last features for next prediction
            # In production, you'd update features based on predictions
            
        return np.array(predictions)

class LSTMPredictor(StockPredictor):
    """LSTM-based stock predictor."""
    
    def __init__(self, ticker: str, sequence_length: int = 60):
        super().__init__(ticker)
        self.sequence_length = sequence_length
        self.model = None
        
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture."""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """Train LSTM model."""
        # Prepare features
        features = self.prepare_features(df)
        
        # Create target
        target = df['close'].shift(-1) / df['close'] - 1
        
        # Remove NaN values
        mask = ~(target.isna() | features.isna().any(axis=1))
        X = features[mask].values
        y = target[mask].values
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y)
        
        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build and train model
        self.model = self.build_model((self.sequence_length, X.shape[1]))
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate metrics
        val_predictions = self.model.predict(X_val)
        self.metrics = {
            'rmse': np.sqrt(np.mean((val_predictions.flatten() - y_val) ** 2)),
            'mae': np.mean(np.abs(val_predictions.flatten() - y_val)),
            'val_loss': min(history.history['val_loss'])
        }
        
        return self.metrics
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> np.ndarray:
        """Make predictions for future days."""
        features = self.prepare_features(df)
        X_scaled = self.scaler.transform(features.values)
        
        # Get last sequence
        last_sequence = X_scaled[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.sequence_length, -1)
            pred = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence (simplified - in production would update all features)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            
        return np.array(predictions)