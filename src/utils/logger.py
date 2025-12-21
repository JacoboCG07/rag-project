"""
Centralized logging module for the RAG project
Uses Loguru for logging with sinks to console and MongoDB
"""
from typing import Optional, Dict, Any
from loguru import logger
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
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
    
    # Console sink - INFO level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # MongoDB sink - INFO level or higher
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
        
        log_document: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record["level"].name,
            "message": record["message"],
            "module": record.get("name", "unknown"),
            "function": record.get("function", "unknown"),
            "line": record.get("line", 0),
            "file": file_name,
            "process_id": process_id,
            "thread_id": thread_id,
            "job_id": job_id,
        }
        
        # Add exception if it exists
        if record.get("exception") is not None:
            exception = record["exception"]
            exception_info = {}
            
            # Extract exception type
            if hasattr(exception, "type") and exception.type:
                exception_info["type"] = exception.type.__name__
            else:
                exception_info["type"] = None
            
            # Extract exception value
            if hasattr(exception, "value") and exception.value:
                exception_info["value"] = str(exception.value)
            else:
                exception_info["value"] = None
            
            # Extract traceback (convert to string if necessary)
            if hasattr(exception, "traceback") and exception.traceback:
                # Traceback can be a Traceback object, convert it to string
                if hasattr(exception.traceback, "format"):
                    exception_info["traceback"] = exception.traceback.format()
                else:
                    exception_info["traceback"] = str(exception.traceback)
            else:
                exception_info["traceback"] = None
            
            log_document["exception"] = exception_info
        
        # Add additional fields if they exist, excluding job_id
        if extra_dict and isinstance(extra_dict, dict):
            # Create a copy without job_id so it doesn't appear duplicated
            extra_without_job_id = {k: v for k, v in extra_dict.items() if k != "job_id"}
            if extra_without_job_id:  # Only add if there are other fields besides job_id
                log_document["extra"] = extra_without_job_id
        
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
        
        # Add MongoDB sink with INFO level or higher
        logger.add(
            _mongodb_sink,
            level="INFO",
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
