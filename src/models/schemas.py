from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum

class FilingType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    TWENTY_F = "20-F"

class IngestRequest(BaseModel):
    years: int = Field(default=10, ge=1, le=20)
    filing_types: List[FilingType] = Field(default=[FilingType.TEN_K, FilingType.TEN_Q])
    force_refresh: bool = False

class PredictionRequest(BaseModel):
    prediction_days: int = Field(default=30, ge=1, le=365)
    model_type: str = Field(default="xgboost", pattern="^(xgboost|lstm|ensemble)$")
    include_confidence: bool = True
    use_cached_model: bool = True

class FinancialStatement(BaseModel):
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "revenue": 394328000000,
                "gross_profit": 169148000000,
                "operating_income": 114301000000,
                "net_income": 96995000000,
                "eps": 6.13
            }
        }

class BalanceSheet(BaseModel):
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    shareholders_equity: Optional[float] = None
    total_debt: Optional[float] = None

class CashFlow(BaseModel):
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None

class FinancialData(BaseModel):
    ticker: str
    company_name: str
    cik: str
    filing_type: FilingType
    filing_date: date
    fiscal_year: int
    fiscal_period: str
    income_statement: FinancialStatement
    balance_sheet: BalanceSheet
    cash_flow: CashFlow
    ratios: Dict[str, float]
    narrative_sections: Dict[str, str]
    
class Prediction(BaseModel):
    date: date
    predicted_price: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    direction: str
    change_percent: float

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predictions: List[Prediction]
    model_metrics: Dict[str, float]
    model_type: str
    training_period: str
    generated_at: datetime

class IngestResponse(BaseModel):
    status: str
    ticker: str
    filings_processed: int
    storage_location: str
    processing_time: float
    errors: List[str] = []

class CompanyInfo(BaseModel):
    ticker: str
    name: str
    cik: str
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    data_available: bool
    last_updated: Optional[datetime] = None