import io
import joblib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from src.models.ml_models import XGBoostPredictor, LSTMPredictor
from src.services.storage_service import S3StorageService
from src.utils.logger import logger
from src.models.schemas import Prediction, PredictionResponse

class MLService:
    """Service for ML model training and prediction."""
    
    def __init__(self):
        self.storage_service = S3StorageService()
        self.models = {}
    
    def train_model(
        self,
        ticker: str,
        model_type: str = 'xgboost',
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """Train ML model for stock prediction."""
        try:
            # Check if recent model exists
            if not force_retrain:
                existing_model = self._load_cached_model(ticker, model_type)
                if existing_model:
                    logger.info(f"Using cached model for {ticker}")
                    return existing_model
            
            logger.info(f"Training {model_type} model for {ticker}")
            
            # Get historical data
            stock_data = self._get_historical_data(ticker)
            
            if stock_data.empty:
                raise ValueError(f"No historical data available for {ticker}")
            
            # Get financial data from S3
            financial_data = self.storage_service.retrieve_financial_data(ticker)
            
            # Enhance stock data with financial metrics
            enhanced_data = self._enhance_with_financials(stock_data, financial_data)
            
            # Train model based on type
            if model_type == 'xgboost':
                model = XGBoostPredictor(ticker)
                metrics = model.train(enhanced_data)
            elif model_type == 'lstm':
                model = LSTMPredictor(ticker)
                metrics = model.train(enhanced_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Save model to S3
            model_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            model_buffer.seek(0)
            
            s3_path = self.storage_service.store_model(
                ticker,
                model_buffer.getvalue(),
                model_type
            )
            
            # Cache model
            self.models[f"{ticker}_{model_type}"] = model
            
            return {
                'ticker': ticker,
                'model_type': model_type,
                'metrics': metrics,
                'storage_path': s3_path,
                'trained_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            raise
    
    def predict(
        self,
        ticker: str,
        model_type: str = 'xgboost',
        prediction_days: int = 30,
        include_confidence: bool = True
    ) -> PredictionResponse:
        """Make stock price predictions."""
        try:
            # Load or train model
            model_key = f"{ticker}_{model_type}"
            
            if model_key not in self.models:
                model = self._load_or_train_model(ticker, model_type)
                self.models[model_key] = model
            else:
                model = self.models[model_key]
            
            # Get recent data for prediction
            recent_data = self._get_historical_data(ticker, days=100)
            
            if recent_data.empty:
                raise ValueError(f"No recent data available for {ticker}")
            
            current_price = recent_data['close'].iloc[-1]
            
            # Make predictions
            predictions_raw = model.predict(recent_data, prediction_days)
            
            # Convert to price predictions
            predictions = []
            base_date = recent_data.index[-1]
            
            for i, pred_return in enumerate(predictions_raw):
                pred_date = base_date + timedelta(days=i+1)
                pred_price = current_price * (1 + pred_return)
                
                prediction = Prediction(
                    date=pred_date.date(),
                    predicted_price=round(pred_price, 2),
                    direction='up' if pred_return > 0 else 'down',
                    change_percent=round(pred_return * 100, 2)
                )
                
                # Add confidence intervals if requested
                if include_confidence:
                    std_dev = np.std(predictions_raw[:i+1]) if i > 0 else 0.02
                    prediction.confidence_lower = round(pred_price * (1 - std_dev), 2)
                    prediction.confidence_upper = round(pred_price * (1 + std_dev), 2)
                
                predictions.append(prediction)
            
            return PredictionResponse(
                ticker=ticker,
                current_price=round(current_price, 2),
                predictions=predictions,
                model_metrics=model.metrics,
                model_type=model_type,
                training_period=f"{recent_data.index[0].year}-{recent_data.index[-1].year}",
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error making predictions for {ticker}: {e}")
            raise
    
    def _get_historical_data(self, ticker: str, days: int = 3650) -> pd.DataFrame:
        """Get historical stock data from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = stock.history(start=start_date, end=end_date)
            
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _enhance_with_financials(
        self,
        stock_data: pd.DataFrame,
        financial_data: List
    ) -> pd.DataFrame:
        """Enhance stock data with financial metrics."""
        enhanced = stock_data.copy()
        
        # Add financial metrics as features
        for fin_data in financial_data:
            # Find the closest date in stock data
            filing_date = pd.Timestamp(fin_data.filing_date)
            
            # Add PE ratio if available
            if fin_data.income_statement.eps and fin_data.income_statement.eps > 0:
                closest_idx = stock_data.index.get_indexer([filing_date], method='nearest')[0]
                if closest_idx < len(stock_data):
                    price_at_filing = stock_data.iloc[closest_idx]['close']
                    pe_ratio = price_at_filing / fin_data.income_statement.eps
                    
                    # Forward fill the PE ratio until next filing
                    enhanced.loc[stock_data.index[closest_idx]:, 'pe_ratio'] = pe_ratio
            
            # Add other ratios
            for ratio_name, value in fin_data.ratios.items():
                if value:
                    closest_idx = stock_data.index.get_indexer([filing_date], method='nearest')[0]
                    if closest_idx < len(stock_data):
                        enhanced.loc[stock_data.index[closest_idx]:, ratio_name] = value
        
        # Forward fill financial metrics
        financial_columns = [col for col in enhanced.columns if col not in stock_data.columns]
        enhanced[financial_columns] = enhanced[financial_columns].fillna(method='ffill')
        
        return enhanced
    
    def _load_cached_model(self, ticker: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Load cached model if recent enough."""
        try:
            model_data = self.storage_service.retrieve_latest_model(ticker, model_type)
            
            if model_data:
                model = joblib.load(io.BytesIO(model_data))
                
                # Check if model is recent enough
                if hasattr(model, 'timestamp'):
                    model_age = datetime.now() - datetime.fromisoformat(model.timestamp)
                    if model_age.days < settings.model_refresh_days:
                        return model
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading cached model: {e}")
            return None
    
    def _load_or_train_model(self, ticker: str, model_type: str):
        """Load existing model or train new one."""
        # Try to load from S3
        model_data = self.storage_service.retrieve_latest_model(ticker, model_type)
        
        if model_data:
            model = joblib.load(io.BytesIO(model_data))
            logger.info(f"Loaded existing model for {ticker}")
            return model
        
        # Train new model
        logger.info(f"Training new model for {ticker}")
        result = self.train_model(ticker, model_type)
        
        # Load the newly trained model
        model_data = self.storage_service.retrieve_latest_model(ticker, model_type)
        return joblib.load(io.BytesIO(model_data))