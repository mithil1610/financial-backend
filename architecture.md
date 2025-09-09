financial-backend/
├── src/
│   ├── services/
│   │   ├── ingestion_service.py    # SEC EDGAR data fetching
│   │   ├── processing_service.py   # Data extraction & transformation
│   │   ├── storage_service.py      # AWS S3 operations
│   │   └── ml_service.py          # ML model training & prediction
│   ├── models/
│   │   ├── schemas.py             # Pydantic models
│   │   └── ml_models.py           # ML model definitions
│   ├── utils/
│   │   ├── config.py              # Configuration management
│   │   ├── logger.py              # Logging setup
│   │   └── helpers.py             # Utility functions
│   └── api.py                     # FastAPI application
├── data/
│   ├── raw/                       # Raw SEC filings
│   ├── processed/                 # Processed JSON/Parquet files
│   └── models/                    # Trained ML models
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md