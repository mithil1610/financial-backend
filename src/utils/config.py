import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # AWS
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "financial-data-bucket")
    
    # SEC EDGAR
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "DefaultCompany default@email.com")
    sec_rate_limit: int = int(os.getenv("SEC_RATE_LIMIT", "10"))
    
    # API
    api_key: str = os.getenv("API_KEY", "")
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ML
    model_refresh_days: int = int(os.getenv("MODEL_REFRESH_DAYS", "7"))
    prediction_confidence_level: float = float(os.getenv("PREDICTION_CONFIDENCE_LEVEL", "0.95"))
    
    # Redis
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    
    class Config:
        env_file = ".env"

settings = Settings()