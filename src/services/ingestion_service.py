import requests
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from bs4 import BeautifulSoup
import yfinance as yf
from src.utils.config import settings
from src.utils.logger import logger
from src.utils.helpers import retry_with_backoff, chunk_list, RateLimiter
from src.models.schemas import FilingType

class SECIngestionService:
    """Service for ingesting data from SEC EDGAR."""
    
    BASE_URL = "https://data.sec.gov"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.sec_user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
        self.rate_limiter = RateLimiter(settings.sec_rate_limit, 1.0)
        
    def get_top_100_companies(self) -> List[Dict[str, str]]:
        """Get top 100 US companies by market cap."""
        logger.info("Fetching top 100 companies by market cap")
        
        # Use S&P 500 as proxy for top companies
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        
        companies = []
        for _, row in sp500.head(100).iterrows():
            companies.append({
                'ticker': row['Symbol'],
                'name': row['Security'],
                'cik': self.get_cik(row['Symbol'])
            })
        
        return companies
    
    @retry_with_backoff(retries=3)
    def get_cik(self, ticker: str) -> str:
        """Get CIK for a given ticker."""
        url = f"{self.BASE_URL}/submissions/CIK{ticker.upper()}.json"
        
        try:
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get('cik', '').zfill(10)
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
        
        # Fallback: try ticker lookup
        return self._lookup_cik(ticker)
    
    def _lookup_cik(self, ticker: str) -> str:
        """Lookup CIK using ticker mapping."""
        try:
            tickers_url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self.session.get(tickers_url)
            tickers_data = response.json()
            
            for item in tickers_data.values():
                if item.get('ticker') == ticker.upper():
                    return str(item.get('cik_str', '')).zfill(10)
        except Exception as e:
            logger.error(f"Error in CIK lookup for {ticker}: {e}")
        
        return ""
    
    @retry_with_backoff(retries=3)
    def get_company_filings(
        self,
        cik: str,
        filing_types: List[str],
        years: int = 10
    ) -> List[Dict[str, Any]]:
        """Get company filings from SEC EDGAR."""
        cik = cik.zfill(10)
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            filings = []
            recent_filings = data.get('filings', {}).get('recent', {})
            
            # Get filing details
            forms = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_documents = recent_filings.get('primaryDocument', [])
            
            cutoff_date = datetime.now() - timedelta(days=years * 365)
            
            for i in range(len(forms)):
                if forms[i] in filing_types:
                    filing_date = datetime.strptime(filing_dates[i], '%Y-%m-%d')
                    
                    if filing_date >= cutoff_date:
                        filings.append({
                            'form_type': forms[i],
                            'filing_date': filing_dates[i],
                            'accession_number': accession_numbers[i].replace('-', ''),
                            'primary_document': primary_documents[i],
                            'cik': cik,
                            'url': self._construct_filing_url(cik, accession_numbers[i], primary_documents[i])
                        })
            
            logger.info(f"Found {len(filings)} filings for CIK {cik}")
            return filings
            
        except Exception as e:
            logger.error(f"Error fetching filings for CIK {cik}: {e}")
            return []
    
    def _construct_filing_url(self, cik: str, accession_number: str, document: str) -> str:
        """Construct URL for filing document."""
        acc_no_dash = accession_number.replace('-', '')
        return f"{self.ARCHIVES_URL}/data/{cik}/{acc_no_dash}/{document}"
    
    @retry_with_backoff(retries=3)
    def download_filing(self, filing: Dict[str, Any]) -> Optional[str]:
        """Download filing content."""
        try:
            response = self.session.get(filing['url'])
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(1 / settings.sec_rate_limit)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error downloading filing {filing['accession_number']}: {e}")
            return None
    
    def ingest_company_data(
        self,
        ticker: str,
        years: int = 10,
        filing_types: List[str] = None
    ) -> Dict[str, Any]:
        """Ingest all financial data for a company."""
        if filing_types is None:
            filing_types = ['10-K', '10-Q']
        
        logger.info(f"Starting ingestion for {ticker}")
        start_time = time.time()
        
        # Get CIK
        cik = self.get_cik(ticker)
        if not cik:
            raise ValueError(f"Could not find CIK for ticker {ticker}")
        
        # Get filings
        filings = self.get_company_filings(cik, filing_types, years)
        
        # Download filings in parallel with rate limiting
        downloaded_filings = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_filing = {
                executor.submit(self.download_filing, filing): filing
                for filing in filings
            }
            
            for future in as_completed(future_to_filing):
                filing = future_to_filing[future]
                try:
                    content = future.result()
                    if content:
                        filing['content'] = content
                        downloaded_filings.append(filing)
                except Exception as e:
                    logger.error(f"Error processing filing: {e}")
        
        # Get additional market data from yfinance
        market_data = self._get_market_data(ticker)
        
        elapsed_time = time.time() - start_time
        
        return {
            'ticker': ticker,
            'cik': cik,
            'filings': downloaded_filings,
            'market_data': market_data,
            'ingestion_time': elapsed_time,
            'filing_count': len(downloaded_filings)
        }
    
    def _get_market_data(self, ticker: str) -> Dict[str, Any]:
        """Get additional market data from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get historical price data
            hist = stock.history(period="10y")
            
            return {
                'current_price': info.get('currentPrice'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'historical_prices': hist.to_dict() if not hist.empty else None
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return {}