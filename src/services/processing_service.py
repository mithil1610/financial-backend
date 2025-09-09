import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from src.utils.logger import logger
from src.utils.helpers import normalize_financial_value, calculate_financial_ratios
from src.models.schemas import FinancialData, FinancialStatement, BalanceSheet, CashFlow

class FinancialDataProcessor:
    """Service for processing raw financial filings into structured data."""
    
    def __init__(self):
        self.xbrl_namespaces = {
            'us-gaap': 'http://fasb.org/us-gaap/',
            'dei': 'http://xbrl.sec.gov/dei/',
            'xbrli': 'http://www.xbrl.org/2003/instance'
        }
        
    def process_filing(self, filing: Dict[str, Any]) -> Optional[FinancialData]:
        """Process a single filing into structured financial data."""
        try:
            content = filing.get('content', '')
            
            # Determine filing format and process accordingly
            if self._is_xbrl(content):
                return self._process_xbrl_filing(filing)
            elif self._is_html(content):
                return self._process_html_filing(filing)
            else:
                return self._process_text_filing(filing)
                
        except Exception as e:
            logger.error(f"Error processing filing {filing.get('accession_number')}: {e}")
            return None
    
    def _is_xbrl(self, content: str) -> bool:
        """Check if content is XBRL format."""
        return 'xbrl' in content.lower()[:1000] or '<xbrl' in content.lower()[:1000]
    
    def _is_html(self, content: str) -> bool:
        """Check if content is HTML format."""
        return '<html' in content.lower()[:500] or '<!DOCTYPE html' in content.lower()[:500]
    
    def _process_xbrl_filing(self, filing: Dict[str, Any]) -> Optional[FinancialData]:
        """Process XBRL formatted filing."""
        try:
            soup = BeautifulSoup(filing['content'], 'lxml-xml')
            
            # Extract financial data from XBRL tags
            financial_data = {
                'income_statement': self._extract_income_statement_xbrl(soup),
                'balance_sheet': self._extract_balance_sheet_xbrl(soup),
                'cash_flow': self._extract_cash_flow_xbrl(soup)
            }
            
            # Extract narrative sections
            narrative = self._extract_narrative_sections(filing['content'])
            
            # Calculate ratios
            ratios = calculate_financial_ratios(
                financial_data['income_statement'],
                financial_data['balance_sheet'],
                financial_data['cash_flow']
            )
            
            return self._create_financial_data_object(filing, financial_data, narrative, ratios)
            
        except Exception as e:
            logger.error(f"Error processing XBRL filing: {e}")
            return None
    
    def _process_html_filing(self, filing: Dict[str, Any]) -> Optional[FinancialData]:
        """Process HTML formatted filing."""
        try:
            soup = BeautifulSoup(filing['content'], 'html.parser')
            
            # Find financial tables
            tables = soup.find_all('table')
            
            financial_data = {
                'income_statement': self._extract_income_statement_html(tables),
                'balance_sheet': self._extract_balance_sheet_html(tables),
                'cash_flow': self._extract_cash_flow_html(tables)
            }
            
            # Extract narrative sections
            narrative = self._extract_narrative_sections_html(soup)
            
            # Calculate ratios
            ratios = calculate_financial_ratios(
                financial_data['income_statement'],
                financial_data['balance_sheet'],
                financial_data['cash_flow']
            )
            
            return self._create_financial_data_object(filing, financial_data, narrative, ratios)
            
        except Exception as e:
            logger.error(f"Error processing HTML filing: {e}")
            return None
    
    def _process_text_filing(self, filing: Dict[str, Any]) -> Optional[FinancialData]:
        """Process text formatted filing."""
        try:
            content = filing['content']
            
            # Use regex patterns to extract financial data
            financial_data = {
                'income_statement': self._extract_income_statement_text(content),
                'balance_sheet': self._extract_balance_sheet_text(content),
                'cash_flow': self._extract_cash_flow_text(content)
            }
            
            # Extract narrative sections
            narrative = self._extract_narrative_sections(content)
            
            # Calculate ratios
            ratios = calculate_financial_ratios(
                financial_data['income_statement'],
                financial_data['balance_sheet'],
                financial_data['cash_flow']
            )
            
            return self._create_financial_data_object(filing, financial_data, narrative, ratios)
            
        except Exception as e:
            logger.error(f"Error processing text filing: {e}")
            return None
    
    def _extract_income_statement_xbrl(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Extract income statement data from XBRL."""
        income_statement = {}
        
        # Common XBRL tags for income statement items
        tag_mappings = {
            'revenue': ['Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomer'],
            'gross_profit': ['GrossProfit'],
            'operating_income': ['OperatingIncomeLoss'],
            'net_income': ['NetIncomeLoss', 'ProfitLoss'],
            'eps': ['EarningsPerShareBasic', 'EarningsPerShareDiluted']
        }
        
        for key, tags in tag_mappings.items():
            for tag in tags:
                element = soup.find(tag)
                if element:
                    income_statement[key] = normalize_financial_value(element.text)
                    break
        
        return income_statement
    
    def _extract_balance_sheet_xbrl(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Extract balance sheet data from XBRL."""
        balance_sheet = {}
        
        tag_mappings = {
            'total_assets': ['Assets'],
            'current_assets': ['AssetsCurrent'],
            'total_liabilities': ['Liabilities'],
            'current_liabilities': ['LiabilitiesCurrent'],
            'shareholders_equity': ['StockholdersEquity'],
            'total_debt': ['LongTermDebt', 'DebtCurrent']
        }
        
        for key, tags in tag_mappings.items():
            for tag in tags:
                element = soup.find(tag)
                if element:
                    balance_sheet[key] = normalize_financial_value(element.text)
                    break
        
        return balance_sheet
    
    def _extract_cash_flow_xbrl(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Extract cash flow data from XBRL."""
        cash_flow = {}
        
        tag_mappings = {
            'operating_cash_flow': ['NetCashProvidedByUsedInOperatingActivities'],
            'investing_cash_flow': ['NetCashProvidedByUsedInInvestingActivities'],
            'financing_cash_flow': ['NetCashProvidedByUsedInFinancingActivities'],
            'free_cash_flow': ['FreeCashFlow']
        }
        
        for key, tags in tag_mappings.items():
            for tag in tags:
                element = soup.find(tag)
                if element:
                    cash_flow[key] = normalize_financial_value(element.text)
                    break
        
        return cash_flow
    
    def _extract_income_statement_html(self, tables: List) -> Dict[str, float]:
        """Extract income statement from HTML tables."""
        income_statement = {}
        
        # Pattern matching for income statement items
        patterns = {
            'revenue': r'(?:total\s+)?(?:net\s+)?(?:revenues?|sales)',
            'gross_profit': r'gross\s+(?:profit|margin)',
            'operating_income': r'operating\s+(?:income|profit)',
            'net_income': r'net\s+(?:income|earnings)',
            'eps': r'earnings?\s+per\s+share'
        }
        
        for table in tables:
            table_text = table.get_text().lower()
            if 'income' in table_text or 'operations' in table_text:
                df = pd.read_html(str(table))[0]
                
                for key, pattern in patterns.items():
                    for idx, row in df.iterrows():
                        row_text = str(row[0]).lower() if not pd.isna(row[0]) else ''
                        if re.search(pattern, row_text):
                            # Try to find the value in subsequent columns
                            for col in range(1, len(row)):
                                val = normalize_financial_value(row[col])
                                if val:
                                    income_statement[key] = val
                                    break
        
        return income_statement
    
    def _extract_balance_sheet_html(self, tables: List) -> Dict[str, float]:
        """Extract balance sheet from HTML tables."""
        balance_sheet = {}
        
        patterns = {
            'total_assets': r'total\s+assets',
            'current_assets': r'(?:total\s+)?current\s+assets',
            'total_liabilities': r'total\s+liabilities',
            'current_liabilities': r'(?:total\s+)?current\s+liabilities',
            'shareholders_equity': r'(?:total\s+)?(?:stockholders?|shareholders?)\s+equity',
            'total_debt': r'(?:total\s+)?(?:long[\s-]?term\s+)?debt'
        }
        
        for table in tables:
            table_text = table.get_text().lower()
            if 'balance' in table_text or 'assets' in table_text:
                df = pd.read_html(str(table))[0]
                
                for key, pattern in patterns.items():
                    for idx, row in df.iterrows():
                        row_text = str(row[0]).lower() if not pd.isna(row[0]) else ''
                        if re.search(pattern, row_text):
                            for col in range(1, len(row)):
                                val = normalize_financial_value(row[col])
                                if val:
                                    balance_sheet[key] = val
                                    break
        
        return balance_sheet
    
    def _extract_cash_flow_html(self, tables: List) -> Dict[str, float]:
        """Extract cash flow from HTML tables."""
        cash_flow = {}
        
        patterns = {
            'operating_cash_flow': r'(?:net\s+)?cash\s+(?:provided\s+by|from)\s+operating',
            'investing_cash_flow': r'(?:net\s+)?cash\s+(?:used\s+in|from)\s+investing',
            'financing_cash_flow': r'(?:net\s+)?cash\s+(?:used\s+in|from)\s+financing',
            'free_cash_flow': r'free\s+cash\s+flow'
        }
        
        for table in tables:
            table_text = table.get_text().lower()
            if 'cash' in table_text and 'flow' in table_text:
                df = pd.read_html(str(table))[0]
                
                for key, pattern in patterns.items():
                    for idx, row in df.iterrows():
                        row_text = str(row[0]).lower() if not pd.isna(row[0]) else ''
                        if re.search(pattern, row_text):
                            for col in range(1, len(row)):
                                val = normalize_financial_value(row[col])
                                if val:
                                    cash_flow[key] = val
                                    break
        
        return cash_flow
    
    def _extract_income_statement_text(self, content: str) -> Dict[str, float]:
        """Extract income statement from text content."""
        income_statement = {}
        
        # Define regex patterns for text extraction
        patterns = {
            'revenue': r'(?:total\s+)?(?:net\s+)?(?:revenues?|sales)[:\s]+\$?([\d,\.]+)',
            'gross_profit': r'gross\s+(?:profit|margin)[:\s]+\$?([\d,\.]+)',
            'operating_income': r'operating\s+(?:income|profit)[:\s]+\$?([\d,\.]+)',
            'net_income': r'net\s+(?:income|earnings)[:\s]+\$?([\d,\.]+)',
            'eps': r'earnings?\s+per\s+share[:\s]+\$?([\d,\.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                income_statement[key] = normalize_financial_value(match.group(1))
        
        return income_statement
    
    def _extract_balance_sheet_text(self, content: str) -> Dict[str, float]:
        """Extract balance sheet from text content."""
        balance_sheet = {}
        
        patterns = {
            'total_assets': r'total\s+assets[:\s]+\$?([\d,\.]+)',
            'current_assets': r'current\s+assets[:\s]+\$?([\d,\.]+)',
            'total_liabilities': r'total\s+liabilities[:\s]+\$?([\d,\.]+)',
            'current_liabilities': r'current\s+liabilities[:\s]+\$?([\d,\.]+)',
            'shareholders_equity': r'(?:stockholders?|shareholders?)\s+equity[:\s]+\$?([\d,\.]+)',
            'total_debt': r'(?:total\s+)?debt[:\s]+\$?([\d,\.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                balance_sheet[key] = normalize_financial_value(match.group(1))
        
        return balance_sheet
    
    def _extract_cash_flow_text(self, content: str) -> Dict[str, float]:
        """Extract cash flow from text content."""
        cash_flow = {}
        
        patterns = {
            'operating_cash_flow': r'cash\s+from\s+operating[:\s]+\$?([\d,\.]+)',
            'investing_cash_flow': r'cash\s+from\s+investing[:\s]+\$?([\d,\.]+)',
            'financing_cash_flow': r'cash\s+from\s+financing[:\s]+\$?([\d,\.]+)',
            'free_cash_flow': r'free\s+cash\s+flow[:\s]+\$?([\d,\.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                cash_flow[key] = normalize_financial_value(match.group(1))
        
        return cash_flow
    
    def _extract_narrative_sections(self, content: str) -> Dict[str, str]:
        """Extract narrative sections from filing."""
        narrative = {}
        
        # MD&A section
        mda_pattern = r"(?:management['s]?\s+discussion\s+and\s+analysis)(.*?)(?:item\s+\d|$)"
        mda_match = re.search(mda_pattern, content, re.IGNORECASE | re.DOTALL)
        if mda_match:
            narrative['mda'] = self._clean_text(mda_match.group(1))[:5000]  # Limit length
        
        # Risk Factors
        risk_pattern = r"(?:risk\s+factors)(.*?)(?:item\s+\d|$)"
        risk_match = re.search(risk_pattern, content, re.IGNORECASE | re.DOTALL)
        if risk_match:
            narrative['risk_factors'] = self._clean_text(risk_match.group(1))[:5000]
        
        # Business Overview
        business_pattern = r"(?:business\s+overview|our\s+business)(.*?)(?:item\s+\d|$)"
        business_match = re.search(business_pattern, content, re.IGNORECASE | re.DOTALL)
        if business_match:
            narrative['business_overview'] = self._clean_text(business_match.group(1))[:3000]
        
        return narrative
    
    def _extract_narrative_sections_html(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract narrative sections from HTML filing."""
        narrative = {}
        
        # Find sections by headers
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        
        for header in headers:
            header_text = header.get_text().lower()
            
            if 'management' in header_text and 'discussion' in header_text:
                # Extract MD&A
                content = self._extract_section_content(header)
                narrative['mda'] = self._clean_text(content)[:5000]
                
            elif 'risk' in header_text and 'factor' in header_text:
                # Extract Risk Factors
                content = self._extract_section_content(header)
                narrative['risk_factors'] = self._clean_text(content)[:5000]
                
            elif 'business' in header_text:
                # Extract Business Overview
                content = self._extract_section_content(header)
                narrative['business_overview'] = self._clean_text(content)[:3000]
        
        return narrative
    
    def _extract_section_content(self, header_element) -> str:
        """Extract content following a header element."""
        content = []
        for sibling in header_element.find_next_siblings():
            if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                break
            content.append(sibling.get_text())
        return ' '.join(content)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        # Trim
        text = text.strip()
        return text
    
    def _create_financial_data_object(
        self,
        filing: Dict[str, Any],
        financial_data: Dict[str, Dict],
        narrative: Dict[str, str],
        ratios: Dict[str, float]
    ) -> FinancialData:
        """Create structured FinancialData object."""
        
        # Parse filing date
        filing_date = datetime.strptime(filing['filing_date'], '%Y-%m-%d').date()
        
        # Determine fiscal year and period
        fiscal_year = filing_date.year
        fiscal_period = 'Q4' if filing['form_type'] == '10-K' else self._determine_quarter(filing_date)
        
        return FinancialData(
            ticker=filing.get('ticker', ''),
            company_name=filing.get('company_name', ''),
            cik=filing['cik'],
            filing_type=filing['form_type'],
            filing_date=filing_date,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            income_statement=FinancialStatement(**financial_data.get('income_statement', {})),
            balance_sheet=BalanceSheet(**financial_data.get('balance_sheet', {})),
            cash_flow=CashFlow(**financial_data.get('cash_flow', {})),
            ratios=ratios,
            narrative_sections=narrative
        )
    
    def _determine_quarter(self, date) -> str:
        """Determine fiscal quarter from date."""
        month = date.month
        if month <= 3:
            return 'Q1'
        elif month <= 6:
            return 'Q2'
        elif month <= 9:
            return 'Q3'
        else:
            return 'Q4'
    
    def convert_to_llm_friendly_format(self, financial_data: FinancialData) -> Dict[str, Any]:
        """Convert financial data to LLM-friendly format."""
        return {
            "metadata": {
                "ticker": financial_data.ticker,
                "company_name": financial_data.company_name,
                "cik": financial_data.cik,
                "filing_type": financial_data.filing_type,
                "filing_date": financial_data.filing_date.isoformat(),
                "fiscal_year": financial_data.fiscal_year,
                "fiscal_period": financial_data.fiscal_period
            },
            "financial_statements": {
                "income_statement": {
                    "revenue": {
                        "value": financial_data.income_statement.revenue,
                        "label": "Total Revenue",
                        "unit": "USD"
                    },
                    "gross_profit": {
                        "value": financial_data.income_statement.gross_profit,
                        "label": "Gross Profit",
                        "unit": "USD"
                    },
                    "operating_income": {
                        "value": financial_data.income_statement.operating_income,
                        "label": "Operating Income",
                        "unit": "USD"
                    },
                    "net_income": {
                        "value": financial_data.income_statement.net_income,
                        "label": "Net Income",
                        "unit": "USD"
                    },
                    "eps": {
                        "value": financial_data.income_statement.eps,
                        "label": "Earnings Per Share",
                        "unit": "USD per share"
                    }
                },
                "balance_sheet": {
                    "total_assets": {
                        "value": financial_data.balance_sheet.total_assets,
                        "label": "Total Assets",
                        "unit": "USD"
                    },
                    "total_liabilities": {
                        "value": financial_data.balance_sheet.total_liabilities,
                        "label": "Total Liabilities",
                        "unit": "USD"
                    },
                    "shareholders_equity": {
                        "value": financial_data.balance_sheet.shareholders_equity,
                        "label": "Shareholders' Equity",
                        "unit": "USD"
                    }
                },
                "cash_flow": {
                    "operating": {
                        "value": financial_data.cash_flow.operating_cash_flow,
                        "label": "Operating Cash Flow",
                        "unit": "USD"
                    },
                    "investing": {
                        "value": financial_data.cash_flow.investing_cash_flow,
                        "label": "Investing Cash Flow",
                        "unit": "USD"
                    },
                    "financing": {
                        "value": financial_data.cash_flow.financing_cash_flow,
                        "label": "Financing Cash Flow",
                        "unit": "USD"
                    }
                }
            },
            "financial_ratios": {
                ratio_name: {
                    "value": value,
                    "label": self._get_ratio_label(ratio_name)
                }
                for ratio_name, value in financial_data.ratios.items()
            },
            "narrative_analysis": financial_data.narrative_sections
        }
    
    def _get_ratio_label(self, ratio_name: str) -> str:
        """Get human-readable label for ratio."""
        labels = {
            'profit_margin': 'Profit Margin',
            'roa': 'Return on Assets',
            'roe': 'Return on Equity',
            'current_ratio': 'Current Ratio',
            'debt_to_equity': 'Debt to Equity Ratio',
            'asset_turnover': 'Asset Turnover Ratio'
        }
        return labels.get(ratio_name, ratio_name.replace('_', ' ').title())