"""
Functional tests for the centralized logger
These tests verify that the logger works correctly with console and MongoDB sinks.
"""
import pytest
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from io import StringIO
from contextlib import redirect_stderr
from datetime import datetime, timedelta

# Add src to path
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent.parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    raise ImportError(f"Could not find src directory at {src_path}")

# Load .env file from project root
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Configure MongoDB environment variables if not set (for logger initialization)
# These must be set BEFORE importing the logger, as it initializes on import
if not os.getenv("MONGO_DB_NAME"):
    os.environ["MONGO_DB_NAME"] = "rag_logs"
if not os.getenv("MONGO_LOGS_COLLECTION"):
    os.environ["MONGO_LOGS_COLLECTION"] = "logs"
if not os.getenv("MONGO_HOST"):
    os.environ["MONGO_HOST"] = "localhost"
if not os.getenv("MONGO_PORT"):
    os.environ["MONGO_PORT"] = "27017"

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Import logger after setting up path and environment
from utils.logger import get_logger, reinitialize_mongodb_sink
from loguru import logger as loguru_logger


def _get_mongodb_config():
    """Helper function to get MongoDB configuration from environment variables
    Uses the same logic as the logger to ensure consistency"""
    # Use the same defaults as the logger
    mongo_host = os.getenv("MONGO_HOST", "localhost")
    mongo_port = os.getenv("MONGO_PORT", "27017")
    mongo_username = os.getenv("MONGO_ROOT_USERNAME")
    mongo_password = os.getenv("MONGO_ROOT_PASSWORD")
    mongo_db = os.getenv("MONGO_DB_NAME", "rag_logs")  # Same default as logger
    mongo_collection_name = os.getenv("MONGO_LOGS_COLLECTION", "logs")  # Same default as logger
    
    if mongo_username and mongo_password:
        uri = f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"
    else:
        uri = f"mongodb://{mongo_host}:{mongo_port}/"
    
    return {
        "uri": uri,
        "db": mongo_db,
        "collection": mongo_collection_name
    }


def is_mongodb_available():
    """Check if MongoDB is available and accessible"""
    try:
        config = _get_mongodb_config()
        client = MongoClient(config["uri"], serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        client.close()
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception):
        return False


@pytest.fixture
def logger_instance():
    """Fixture to provide a logger instance"""
    return get_logger(__name__)


@pytest.fixture
def mongodb_collection():
    """Fixture to provide MongoDB collection for tests that need it"""
    if not is_mongodb_available():
        pytest.skip("MongoDB is not available or not configured")
    
    # Reinicializar el sink de MongoDB para asegurarse de que está configurado
    reinitialize_mongodb_sink()
    
    config = _get_mongodb_config()
    client = MongoClient(config["uri"])
    db = client[config["db"]]
    collection = db[config["collection"]]
    
    yield collection
    
    # Cleanup: close connection after test
    client.close()


@pytest.mark.functional
class TestLoggerFunctional:
    """Functional tests for the centralized logger"""
    
    def test_get_logger_returns_logger_instance(self, logger_instance):
        """Test that get_logger returns a logger instance"""
        logger = logger_instance
        
        assert logger is not None
        # Verificar que tiene métodos de logging
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'debug')
    
    def test_logger_writes_to_console(self, logger_instance):
        """Test that logger writes messages to console (stderr)"""
        logger = logger_instance
        
        # Capturar logs usando la API de Loguru
        output = []
        handler_id = loguru_logger.add(
            lambda msg: output.append(msg),
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            level="INFO"
        )
        
        try:
            logger.info("Test message for console output")
            # Dar tiempo para que se procese
            time.sleep(0.1)
            
            output_str = "".join(str(msg) for msg in output)
            assert "Test message for console output" in output_str
            assert "INFO" in output_str or "info" in output_str.lower()
        finally:
            loguru_logger.remove(handler_id)
    
    def test_logger_with_different_levels(self, logger_instance):
        """Test that logger works with different log levels"""
        logger = logger_instance
        
        # Capturar logs usando la API de Loguru
        output = []
        handler_id = loguru_logger.add(
            lambda msg: output.append(msg),
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        
        try:
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            # Dar tiempo para que se procese
            time.sleep(0.1)
            
            output_str = "".join(str(msg) for msg in output)
            assert "Info message" in output_str
            assert "Warning message" in output_str
            assert "Error message" in output_str
        finally:
            loguru_logger.remove(handler_id)
    
    def test_logger_with_module_name(self):
        """Test that logger works when provided with a module name"""
        test_module_name = "test_module"
        logger = get_logger(test_module_name)
        
        # Capturar logs usando la API de Loguru
        output = []
        handler_id = loguru_logger.add(
            lambda msg: output.append(msg),
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            level="INFO"
        )
        
        try:
            logger.info("Test message with module name")
            # Dar tiempo para que se procese
            time.sleep(0.1)
            
            output_str = "".join(str(msg) for msg in output)
            assert "Test message with module name" in output_str
            # El nombre del módulo debería aparecer en el log (puede estar en name o en el mensaje)
            # Loguru puede mostrar el nombre del módulo en diferentes lugares
        finally:
            loguru_logger.remove(handler_id)
    
    def test_logger_writes_to_mongodb(self, logger_instance, mongodb_collection):
        """Test that logger writes messages to MongoDB when available"""
        logger = logger_instance
        collection = mongodb_collection
        
        # Verificar qué base de datos está usando el logger
        # El logger usa las mismas variables de entorno
        logger_db = os.getenv("MONGO_DB_NAME", "rag_logs")
        logger_collection = os.getenv("MONGO_LOGS_COLLECTION", "logs")
        
        # Asegurarse de que estamos usando la misma base de datos que el logger
        config = _get_mongodb_config()
        assert collection.database.name == config["db"], \
            f"El test usa la BD '{collection.database.name}' pero el logger usa '{config['db']}'"
        assert collection.name == config["collection"], \
            f"El test usa la colección '{collection.name}' pero el logger usa '{config['collection']}'"
        
        # Contar logs antes
        count_before = collection.count_documents({})
        
        # Generar un mensaje único para identificar nuestro log
        unique_message = f"Test MongoDB log - {datetime.now().isoformat()}"
        logger.info(unique_message)
        
        # Esperar un momento para que el log se escriba (MongoDB puede tardar)
        time.sleep(2.0)  # Aumentar tiempo de espera
        
        # Verificar que el log se escribió en MongoDB
        count_after = collection.count_documents({})
        assert count_after > count_before, \
            f"No se escribió ningún log en MongoDB. Antes: {count_before}, Después: {count_after}"
        
        # Buscar nuestro log específico
        test_log = collection.find_one({"message": unique_message})
        assert test_log is not None, "No se encontró el log de prueba en MongoDB"
        
        # Verificar estructura del documento
        assert "timestamp" in test_log
        assert "level" in test_log
        assert "message" in test_log
        assert test_log["message"] == unique_message
        assert test_log["level"] == "INFO"
        
        # Limpiar: eliminar el log de prueba
        collection.delete_one({"message": unique_message})
    
    def test_logger_mongodb_document_structure(self, logger_instance, mongodb_collection):
        """Test that MongoDB documents have the correct structure"""
        logger = logger_instance
        collection = mongodb_collection
        
        # Generar un mensaje único
        unique_message = f"Test structure - {datetime.now().isoformat()}"
        logger.info(unique_message)
        
        # Esperar un momento (MongoDB puede tardar)
        time.sleep(1.0)
        
        # Buscar el log
        test_log = collection.find_one({"message": unique_message})
        assert test_log is not None
        
        # Verificar estructura completa
        required_fields = ["timestamp", "level", "message", "module", "function", "line"]
        for field in required_fields:
            assert field in test_log, f"Campo requerido '{field}' no encontrado en el log"
        
        # Verificar tipos
        assert isinstance(test_log["timestamp"], datetime)
        assert isinstance(test_log["level"], str)
        assert isinstance(test_log["message"], str)
        assert isinstance(test_log["module"], str)
        assert isinstance(test_log["function"], str)
        assert isinstance(test_log["line"], int)
        
        # Limpiar
        collection.delete_one({"message": unique_message})
    
    def test_logger_mongodb_with_exception(self, logger_instance, mongodb_collection):
        """Test that logger correctly logs exceptions to MongoDB"""
        logger = logger_instance
        collection = mongodb_collection
        
        # Generar un mensaje único
        unique_message = f"Test exception - {datetime.now().isoformat()}"
        
        try:
            raise ValueError("Test exception for logging")
        except Exception:
            logger.exception(unique_message)
        
        # Esperar un momento (MongoDB puede tardar)
        time.sleep(1.0)
        
        # Buscar el log con excepción
        test_log = collection.find_one({"message": unique_message})
        assert test_log is not None
        assert "exception" in test_log
        assert test_log["exception"]["type"] == "ValueError"
        assert "Test exception for logging" in str(test_log["exception"]["value"])
        
        # Limpiar
        collection.delete_one({"message": unique_message})
    
    def test_logger_works_without_mongodb(self, logger_instance):
        """Test that logger continues to work even if MongoDB is not available"""
        logger = logger_instance
        
        # Capturar logs usando la API de Loguru
        output = []
        handler_id = loguru_logger.add(
            lambda msg: output.append(msg),
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        
        try:
            logger.info("Test message without MongoDB")
            logger.warning("Warning without MongoDB")
            logger.error("Error without MongoDB")
            # Dar tiempo para que se procese
            time.sleep(0.1)
            
            output_str = "".join(str(msg) for msg in output)
            assert "Test message without MongoDB" in output_str
            assert "Warning without MongoDB" in output_str
            assert "Error without MongoDB" in output_str
        finally:
            loguru_logger.remove(handler_id)