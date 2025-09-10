from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import asyncio
from datetime import datetime
from src.services.ingestion_service import SECIngestionService
from src.services.processing_service import FinancialDataProcessor
from src.services.storage_service import S3StorageService
from src.services.ml_service import MLService
from src.models.schemas import (
    IngestRequest,
    IngestResponse,
    PredictionRequest,
    PredictionResponse,
    FinancialData,
    CompanyInfo
)
from src.utils.config import settings
from src.utils.logger import logger
import redis
import json

# Initialize FastAPI app
app = FastAPI(
    title="Financial Data Backend API",
    description="Production-ready backend for financial data ingestion, processing, and ML predictions",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize services
ingestion_service = SECIngestionService()
processor = FinancialDataProcessor()
storage_service = S3StorageService()
ml_service = MLService()

# Initialize Redis cache (optional)
try:
    redis_client = redis.from_url(settings.redis_url) if settings.redis_url else None
except:
    redis_client = None
# redis_client = None

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for authentication."""
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Financial Data Backend API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ingestion": "operational",
            "processing": "operational",
            "storage": "operational",
            "ml": "operational",
            "cache": "operational" if redis_client else "disabled"
        }
    }

@app.get("/companies", response_model=List[CompanyInfo])
async def list_companies(api_key: str = Depends(verify_api_key)):
    """Get list of top 100 companies with available data."""
    try:
        # Check cache
        if redis_client:
            cached = redis_client.get("companies_list")
            if cached:
                return json.loads(cached)
        
        companies = ingestion_service.get_top_100_companies()
        
        # Cache for 1 hour
        if redis_client:
            redis_client.setex(
                "companies_list",
                3600,
                json.dumps(companies)
            )
        
        return companies
        
    except Exception as e:
        logger.error(f"Error listing companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/{ticker}", response_model=IngestResponse)
async def ingest_financial_data(
    ticker: str,
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Ingest financial data for a specific ticker."""
    try:
        ticker = ticker.upper()
        
        # Start ingestion
        start_time = datetime.now()
        
        # Ingest data
        ingested_data = ingestion_service.ingest_company_data(
            ticker,
            request.years,
            request.filing_types
        )
        
        # Process filings
        processed_count = 0
        errors = []
        
        for filing in ingested_data['filings']:
            try:
                # Process filing
                financial_data = processor.process_filing(filing)
                
                if financial_data:
                    # Store in S3
                    storage_path = storage_service.store_financial_data(
                        ticker,
                        financial_data
                    )
                    processed_count += 1
                    
            except Exception as e:
                errors.append(f"Error processing filing {filing.get('accession_number')}: {str(e)}")
                logger.error(f"Processing error: {e}")
        
        # Trigger model training in background
        if request.force_refresh:
            background_tasks.add_task(
                ml_service.train_model,
                ticker,
                'xgboost',
                True
            )
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        return IngestResponse(
            status="success" if processed_count > 0 else "partial",
            ticker=ticker,
            filings_processed=processed_count,
            storage_location=f"s3://{settings.s3_bucket_name}/processed/{ticker}/",
            processing_time=elapsed_time,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error ingesting data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-financials/{ticker}")
async def get_financial_data(
    ticker: str,
    year: Optional[int] = None,
    filing_type: Optional[str] = None,
    format: str = "json",
    api_key: str = Depends(verify_api_key)
):
    """Retrieve processed financial data."""
    try:
        ticker = ticker.upper()
        
        # Check cache
        cache_key = f"financials:{ticker}:{year}:{filing_type}"
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Retrieve from S3
        financial_data = storage_service.retrieve_financial_data(
            ticker,
            year,
            filing_type
        )
        
        if not financial_data:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        # Convert to LLM-friendly format if requested
        if format == "llm":
            result = [
                processor.convert_to_llm_friendly_format(data)
                for data in financial_data
            ]
        else:
            result = [data.dict() for data in financial_data]
        
        # Cache for 15 minutes
        if redis_client:
            redis_client.setex(cache_key, 900, json.dumps(result))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving financial data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{ticker}", response_model=PredictionResponse)
async def predict_stock_price(
    ticker: str,
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate stock price predictions."""
    try:
        ticker = ticker.upper()
        
        # Check cache
        cache_key = f"prediction:{ticker}:{request.model_type}:{request.prediction_days}"
        if redis_client and request.use_cached_model:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Generate predictions
        predictions = ml_service.predict(
            ticker,
            request.model_type,
            request.prediction_days,
            request.include_confidence
        )
        
        # Cache for 5 minutes
        if redis_client:
            redis_client.setex(
                cache_key,
                300,
                predictions.json()
            )
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error generating predictions for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model/{ticker}")
async def train_model(
    ticker: str,
    model_type: str = "xgboost",
    force_retrain: bool = False,
    api_key: str = Depends(verify_api_key)
):
    """Train ML model for a specific ticker."""
    try:
        ticker = ticker.upper()
        
        result = ml_service.train_model(
            ticker,
            model_type,
            force_retrain
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error training model for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )