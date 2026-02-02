"""
Centralized logging module for the RAG project
Uses Loguru for logging with sinks to console and MongoDB
"""

import os
import sys
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Optional, Dict, Any
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load environment variables
load_dotenv()

# Global variable to control initialization
_logger_initialized = False
_mongo_client: Optional[MongoClient] = None
_mongo_collection = None


def _initialize_logger() -> None:
    """
    Initializes the logger with configured sinks.
    This function only runs once.
    """
    global _logger_initialized
    
    if _logger_initialized:
        return
    
    # Remove Loguru's default handler
    logger.remove()
    
    # Console sink - INFO level (shows INFO and above in console)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # MongoDB sink - configurable level (default: WARNING, only warnings and errors)
    # Set MONGO_LOG_LEVEL env var to change: DEBUG, INFO, WARNING, ERROR, CRITICAL
    _setup_mongodb_sink()
    _logger_initialized = True


def _mongodb_sink(message: Any) -> None:
    """
    Custom sink to write logs to MongoDB.
    
    Args:
        message: Loguru message object with all log information
    """
    global _mongo_collection
    
    # If collection is not configured, try to configure it again
    if _mongo_collection is None:
        _setup_mongodb_sink()
        # If still None after trying to configure, exit
        if _mongo_collection is None:
            return
    
    try:
        # DEBUG: Verify we have the collection
        if _mongo_collection is None:
            return
        # Extract information from Loguru message
        record = message.record
        
        # Create document for MongoDB
        # Loguru timestamp is already a datetime object
        timestamp = record["time"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Extract process_id and thread_id correctly
        process_id = 0
        if record.get("process"):
            process = record["process"]
            if hasattr(process, "id"):
                process_id = process.id
            elif isinstance(process, dict):
                process_id = process.get("id", 0)
            else:
                process_id = int(process) if process else 0
        
        thread_id = 0
        if record.get("thread"):
            thread = record["thread"]
            if hasattr(thread, "id"):
                thread_id = thread.id
            elif isinstance(thread, dict):
                thread_id = thread.get("id", 0)
            else:
                thread_id = int(thread) if thread else 0
        
        # Extract job_id from record["extra"] (when using logger.bind(job_id=...))
        job_id = None
        extra_dict = record.get("extra")
        if extra_dict and isinstance(extra_dict, dict):
            job_id = extra_dict.get("job_id")
        
        # Extract file name correctly
        file_name = "unknown"
        if record.get("file"):
            file_obj = record["file"]
            if hasattr(file_obj, "name"):
                file_name = file_obj.name
            elif isinstance(file_obj, dict):
                file_name = file_obj.get("name", "unknown")
            else:
                file_name = str(file_obj)
        
        # Group code-related information in a nested dictionary
        code_info: Dict[str, Any] = {
            "function": record.get("function", "unknown"),
            "module": record.get("name", "unknown"),
            "file": file_name,
            "line": record.get("line", 0),
            "process_id": process_id,
            "thread_id": thread_id
        }
        
        log_document: Dict[str, Any] = {
            "timestamp": timestamp,
            "job_id": job_id,
            "level": record["level"].name,
            "message": record["message"],
            "code_info": code_info
        }
        
        # Add error_info if exception exists
        if record.get("exception") is not None:
            exception = record["exception"]
            error_info = {}
            
            # Extract exception type
            if hasattr(exception, "type") and exception.type:
                error_info["type"] = exception.type.__name__
            else:
                error_info["type"] = None
            
            # Extract exception value
            if hasattr(exception, "value") and exception.value:
                error_info["value"] = str(exception.value)
            else:
                error_info["value"] = None
            
            # Extract traceback (convert to string if necessary)
            if hasattr(exception, "traceback") and exception.traceback:
                # Traceback can be a Traceback object, convert it to string
                if hasattr(exception.traceback, "format"):
                    error_info["traceback"] = exception.traceback.format()
                else:
                    error_info["traceback"] = str(exception.traceback)
            else:
                error_info["traceback"] = None
            
            log_document["error_info"] = error_info
        
        # Add additional fields if they exist, excluding job_id and name (already in code_info)
        if extra_dict and isinstance(extra_dict, dict):
            # Flatten nested extra structure if it exists
            flattened_extra = {}
            for k, v in extra_dict.items():
                # Skip job_id (already at top level)
                if k == "job_id":
                    continue
                # Skip name (already in code_info.module)
                if k == "name":
                    continue
                # If there's a nested "extra" dict, merge its contents
                if k == "extra" and isinstance(v, dict):
                    flattened_extra.update(v)
                else:
                    flattened_extra[k] = v
            
            if flattened_extra:  # Only add if there are fields
                log_document["extra"] = flattened_extra
        
        # Insert into MongoDB
        _mongo_collection.insert_one(log_document)
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        # If connection fails, do nothing to avoid infinite loops
        # The log will be lost but the application will continue working
        # In debug mode, we could log this, but we don't want infinite loops
        pass
    except Exception as e:
        # Catch any other error to prevent the sink from breaking the application
        # Temporarily: write error to stderr for debugging (development only)
        import sys
        if os.getenv("DEBUG_MONGODB_SINK", "false").lower() == "true":
            sys.stderr.write(f"Error in MongoDB sink: {type(e).__name__}: {str(e)}\n")
        pass


def _setup_mongodb_sink() -> None:
    """
    Configures the MongoDB sink to persist logs.
    Reads configuration from environment variables.
    """
    global _mongo_client, _mongo_collection
    
    # Check if MongoDB is disabled for tests
    if os.getenv("DISABLE_MONGODB_LOGGING", "false").lower() == "true":
        _mongo_client = None
        _mongo_collection = None
        return
    
    # Get configuration from environment variables
    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB_NAME", "rag_logs")  # Default value
    mongo_collection_name = os.getenv("MONGO_LOGS_COLLECTION", "logs")  # Default value
    
    # If no URI is configured, build one from individual variables or use default
    if not mongo_uri:
        mongo_host = os.getenv("MONGO_HOST", "localhost")  # Default value
        mongo_port = os.getenv("MONGO_PORT", "27017")  # Default value
        mongo_username = os.getenv("MONGO_ROOT_USERNAME")
        mongo_password = os.getenv("MONGO_ROOT_PASSWORD")
        
        if mongo_username and mongo_password:
            mongo_uri = f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"
        else:
            mongo_uri = f"mongodb://{mongo_host}:{mongo_port}/"
    
    try:
        # Connect to MongoDB
        _mongo_client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=5000
        )
        
        # Verify connection
        _mongo_client.admin.command('ping')
        
        # Get database and collection
        db = _mongo_client[mongo_db]
        _mongo_collection = db[mongo_collection_name]
        
        # Create index on timestamp to improve queries
        _mongo_collection.create_index("timestamp")
        _mongo_collection.create_index("level")
        _mongo_collection.create_index([("timestamp", -1), ("level", 1)])
        # Create indexes on code_info fields for better querying
        _mongo_collection.create_index("code_info.module")
        _mongo_collection.create_index("code_info.process_id")
        _mongo_collection.create_index("code_info.thread_id")
        # Create indexes on error_info fields for better error querying
        _mongo_collection.create_index("error_info.type")
        
        # Get MongoDB log level from environment variable (default: WARNING)
        # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        # WARNING means only warnings and errors are saved to MongoDB
        mongo_log_level = os.getenv("MONGO_LOG_LEVEL", "WARNING").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if mongo_log_level not in valid_levels:
            mongo_log_level = "WARNING"  # Fallback to WARNING if invalid
        
        # Add MongoDB sink with configurable level
        logger.add(
            _mongodb_sink,
            level=mongo_log_level,
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True
        )
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        # If connection fails, simply don't configure the sink
        # The application will continue working without MongoDB logging
        _mongo_client = None
        _mongo_collection = None
    except Exception as e:
        # Any other error is also ignored
        _mongo_client = None
        _mongo_collection = None


def get_logger(name: Optional[str] = None):
    """
    Gets an instance of the configured logger.
    
    Args:
        name: Optional name for the logger (useful for identifying the module)
    
    Returns:
        Logger: Configured Loguru logger instance
    
    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Information message")
    """
    # Ensure logger is initialized
    if not _logger_initialized:
        _initialize_logger()
    
    # If a name is provided, use the logger with that name
    if name:
        return logger.bind(name=name)
    
    return logger


def reinitialize_mongodb_sink():
    """
    Reinitializes the MongoDB sink if it wasn't available during initialization.
    Useful for tests or when MongoDB starts after the logger.
    """
    global _logger_initialized, _mongo_collection
    
    # If MongoDB is not configured, try to configure it now
    if _mongo_collection is None:
        # Remove existing MongoDB sink if there is one
        try:
            # Find and remove MongoDB handlers
            handlers = logger._core.handlers.values()
            for handler_id, handler in list(handlers.items()):
                if hasattr(handler, '_sink') and handler._sink == _mongodb_sink:
                    logger.remove(handler_id)
        except:
            pass
        
        # Try to configure MongoDB again
        _setup_mongodb_sink()


def check_mongodb_connection() -> Dict[str, Any]:
    """
    Checks the status of the MongoDB connection.
    Useful for debugging.
    
    Returns:
        Dict with information about the connection status
    """
    global _mongo_client, _mongo_collection
    
    result = {
        "connected": False,
        "client_exists": _mongo_client is not None,
        "collection_exists": _mongo_collection is not None,
        "error": None
    }
    
    if _mongo_client is None:
        result["error"] = "MongoDB client is not initialized"
        return result
    
    try:
        # Verify connection
        _mongo_client.admin.command('ping')
        result["connected"] = True
        
        if _mongo_collection is not None:
            # Verify we can access the collection
            db_name = _mongo_collection.database.name
            collection_name = _mongo_collection.name
            result["database"] = db_name
            result["collection"] = collection_name
            result["document_count"] = _mongo_collection.count_documents({})
    except Exception as e:
        result["error"] = str(e)
        result["connected"] = False
    
    return result


# Automatically initialize when importing the module
_initialize_logger()
