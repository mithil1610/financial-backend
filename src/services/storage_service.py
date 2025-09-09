import io
import json
import gzip
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from botocore.exceptions import ClientError
from src.utils.config import settings
from src.utils.logger import logger
from src.utils.helpers import calculate_file_hash
from src.models.schemas import FinancialData

class S3StorageService:
    """Service for storing and retrieving data from AWS S3."""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket_name = settings.s3_bucket_name
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure S3 bucket exists, create if not."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"Creating S3 bucket: {self.bucket_name}")
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': settings.aws_region}
                )
    
    def store_financial_data(
        self,
        ticker: str,
        data: FinancialData,
        format: str = 'parquet'
    ) -> str:
        """Store financial data in S3."""
        try:
            # Convert to DataFrame for Parquet storage
            df = self._financial_data_to_dataframe(data)
            
            # Create partition path
            key = self._create_partition_key(ticker, data.fiscal_year, data.filing_type)
            
            if format == 'parquet':
                # Convert to Parquet
                buffer = io.BytesIO()
                table = pa.Table.from_pandas(df)
                pq.write_table(table, buffer, compression='snappy')
                buffer.seek(0)
                
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f"{key}/data.parquet",
                    Body=buffer.getvalue(),
                    ContentType='application/octet-stream',
                    Metadata={
                        'ticker': ticker,
                        'filing_type': data.filing_type,
                        'fiscal_year': str(data.fiscal_year),
                        'created_at': datetime.now().isoformat()
                    }
                )
                
            elif format == 'json':
                # Store as compressed JSON
                json_data = data.dict()
                compressed = gzip.compress(json.dumps(json_data).encode())
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=f"{key}/data.json.gz",
                    Body=compressed,
                    ContentType='application/json',
                    ContentEncoding='gzip'
                )
            
            s3_path = f"s3://{self.bucket_name}/{key}/"
            logger.info(f"Stored financial data for {ticker} at {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Error storing financial data: {e}")
            raise
    
    def retrieve_financial_data(
        self,
        ticker: str,
        year: Optional[int] = None,
        filing_type: Optional[str] = None
    ) -> List[FinancialData]:
        """Retrieve financial data from S3."""
        try:
            # Build prefix for listing objects
            prefix = f"processed/{ticker}/"
            if year:
                prefix += f"year={year}/"
            if filing_type:
                prefix += f"type={filing_type}/"
            
            # List objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.info(f"No data found for {ticker}")
                return []
            
            financial_data_list = []
            
            for obj in response['Contents']:
                if obj['Key'].endswith('.parquet'):
                    # Download and read Parquet file
                    response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    
                    df = pd.read_parquet(io.BytesIO(response['Body'].read()))
                    
                    # Convert DataFrame back to FinancialData
                    for _, row in df.iterrows():
                        financial_data = self._dataframe_to_financial_data(row)
                        financial_data_list.append(financial_data)
                        
                elif obj['Key'].endswith('.json.gz'):
                    # Download and decompress JSON
                    response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    
                    decompressed = gzip.decompress(response['Body'].read())
                    json_data = json.loads(decompressed)
                    
                    financial_data = FinancialData(**json_data)
                    financial_data_list.append(financial_data)
            
            logger.info(f"Retrieved {len(financial_data_list)} records for {ticker}")
            return financial_data_list
            
        except Exception as e:
            logger.error(f"Error retrieving financial data: {e}")
            return []
    
    def store_model(self, ticker: str, model_data: bytes, model_type: str) -> str:
        """Store ML model in S3."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            key = f"models/{ticker}/{model_type}_{timestamp}.pkl"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=model_data,
                ContentType='application/octet-stream',
                Metadata={
                    'ticker': ticker,
                    'model_type': model_type,
                    'created_at': datetime.now().isoformat()
                }
            )
            
            s3_path = f"s3://{self.bucket_name}/{key}"
            logger.info(f"Stored model for {ticker} at {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Error storing model: {e}")
            raise
    
    def retrieve_latest_model(self, ticker: str, model_type: str) -> Optional[bytes]:
        """Retrieve latest ML model from S3."""
        try:
            prefix = f"models/{ticker}/{model_type}_"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return None
            
            # Get the most recent model
            latest = sorted(response['Contents'], key=lambda x: x['LastModified'])[-1]
            
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=latest['Key']
            )
            
            return response['Body'].read()
            
        except Exception as e:
            logger.error(f"Error retrieving model: {e}")
            return None
    
    def _create_partition_key(self, ticker: str, year: int, filing_type: str) -> str:
        """Create S3 partition key."""
        return f"processed/{ticker}/year={year}/type={filing_type}"
    
    def _financial_data_to_dataframe(self, data: FinancialData) -> pd.DataFrame:
        """Convert FinancialData to DataFrame."""
        flat_data = {
            'ticker': data.ticker,
            'company_name': data.company_name,
            'cik': data.cik,
            'filing_type': data.filing_type,
            'filing_date': data.filing_date,
            'fiscal_year': data.fiscal_year,
            'fiscal_period': data.fiscal_period
        }
        
        # Flatten income statement
        for field, value in data.income_statement.dict().items():
            flat_data[f'income_{field}'] = value
        
        # Flatten balance sheet
        for field, value in data.balance_sheet.dict().items():
            flat_data[f'balance_{field}'] = value
        
        # Flatten cash flow
        for field, value in data.cash_flow.dict().items():
            flat_data[f'cash_{field}'] = value
        
        # Add ratios
        for ratio_name, value in data.ratios.items():
            flat_data[f'ratio_{ratio_name}'] = value
        
        return pd.DataFrame([flat_data])
    
    def _dataframe_to_financial_data(self, row: pd.Series) -> FinancialData:
        """Convert DataFrame row to FinancialData."""
        income_data = {}
        balance_data = {}
        cash_data = {}
        ratios = {}
        
        for col, value in row.items():
            if col.startswith('income_'):
                income_data[col.replace('income_', '')] = value
            elif col.startswith('balance_'):
                balance_data[col.replace('balance_', '')] = value
            elif col.startswith('cash_'):
                cash_data[col.replace('cash_', '')] = value
            elif col.startswith('ratio_'):
                ratios[col.replace('ratio_', '')] = value
        
        return FinancialData(
            ticker=row['ticker'],
            company_name=row['company_name'],
            cik=row['cik'],
            filing_type=row['filing_type'],
            filing_date=row['filing_date'],
            fiscal_year=row['fiscal_year'],
            fiscal_period=row['fiscal_period'],
            income_statement=FinancialStatement(**income_data),
            balance_sheet=BalanceSheet(**balance_data),
            cash_flow=CashFlow(**cash_data),
            ratios=ratios,
            narrative_sections={}
        )