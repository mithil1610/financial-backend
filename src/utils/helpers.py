import hashlib
import time
from typing import Any, Callable, Dict, List, Optional
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

def retry_with_backoff(
    retries: int = 3,
    backoff_in_seconds: float = 1,
    exponential: bool = True
) -> Callable:
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = backoff_in_seconds
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        raise e
                    time.sleep(x)
                    if exponential:
                        x *= 2
            return None
        return wrapper
    return decorator

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def normalize_financial_value(value: Any) -> Optional[float]:
    """Normalize financial values to float."""
    if pd.isna(value) or value is None:
        return None
    
    if isinstance(value, str):
        # Remove common formatting
        value = value.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
        
        # Handle millions/billions notation
        multipliers = {'M': 1e6, 'B': 1e9, 'K': 1e3}
        for suffix, multiplier in multipliers.items():
            if value.endswith(suffix):
                return float(value[:-1]) * multiplier
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def calculate_financial_ratios(
    income_statement: Dict[str, float],
    balance_sheet: Dict[str, float],
    cash_flow: Dict[str, float]
) -> Dict[str, Optional[float]]:
    """Calculate common financial ratios."""
    ratios = {}
    
    try:
        # Profitability ratios
        if income_statement.get('revenue') and income_statement.get('net_income'):
            ratios['profit_margin'] = income_statement['net_income'] / income_statement['revenue']
        
        if balance_sheet.get('total_assets') and income_statement.get('net_income'):
            ratios['roa'] = income_statement['net_income'] / balance_sheet['total_assets']
        
        if balance_sheet.get('shareholders_equity') and income_statement.get('net_income'):
            ratios['roe'] = income_statement['net_income'] / balance_sheet['shareholders_equity']
        
        # Liquidity ratios
        if balance_sheet.get('current_assets') and balance_sheet.get('current_liabilities'):
            ratios['current_ratio'] = balance_sheet['current_assets'] / balance_sheet['current_liabilities']
        
        # Leverage ratios
        if balance_sheet.get('total_debt') and balance_sheet.get('shareholders_equity'):
            ratios['debt_to_equity'] = balance_sheet['total_debt'] / balance_sheet['shareholders_equity']
        
        # Efficiency ratios
        if income_statement.get('revenue') and balance_sheet.get('total_assets'):
            ratios['asset_turnover'] = income_statement['revenue'] / balance_sheet['total_assets']
            
    except (ZeroDivisionError, TypeError):
        pass
    
    return ratios

class RateLimiter:
    """Rate limiter for API calls."""
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.timestamps = []
    
    async def acquire(self):
        now = time.time()
        # Remove old timestamps
        self.timestamps = [t for t in self.timestamps if now - t < self.period]
        
        if len(self.timestamps) >= self.calls:
            sleep_time = self.period - (now - self.timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self.timestamps.pop(0)
        
        self.timestamps.append(time.time())