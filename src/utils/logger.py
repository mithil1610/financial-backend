import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
import traceback

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if hasattr(record, "correlation_id"):
            log_obj["correlation_id"] = record.correlation_id
            
        if record.exc_info:
            log_obj["exception"] = traceback.format_exception(*record.exc_info)
            
        return json.dumps(log_obj)

def setup_logger(name: str = __name__, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    
    return logger

logger = setup_logger("financial_backend", settings.log_level if 'settings' in locals() else "INFO")