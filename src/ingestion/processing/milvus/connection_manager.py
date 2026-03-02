"""
Milvus connection manager
Single responsibility: connection and database management
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pymilvus import connections, db
from pymilvus.exceptions import ConnectionNotExistException

from .exceptions import MilvusConnectionError
from src.utils import get_logger

# Load environment variables
load_dotenv(dotenv_path="/code/.env")
load_dotenv(dotenv_path="/code/.env.local", override=True)


class ConnectionManager:
    """
    Manages connections with Milvus.
    Single responsibility: connection handling.
    """

    @staticmethod
    def connect(
        *,
        alias: str = "default",
        uri: Optional[str] = None,
        token: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None
    ) -> None:
        """
        Establishes connection with Milvus.

        Args:
            alias: Connection alias.
            uri: Connection URI (for Milvus Cloud).
            token: Authentication token (for Milvus Cloud).
            host: Milvus host (for local instance).
            port: Milvus port (for local instance).

        Raises:
            MilvusConnectionError: If connection cannot be established.
        """
        logger = get_logger(__name__)
        
        try:
            logger.debug(
                "Connecting to Milvus",
                extra={
                    "alias": alias,
                    "has_uri": bool(uri),
                    "has_token": bool(token),
                    "has_host": bool(host),
                    "has_port": bool(port)
                }
            )
            
            # Try connection with URI (Milvus Cloud)
            if uri and token:
                connections.connect(uri=uri, token=token, alias=alias)
                logger.info("Connected to Milvus using URI and token", extra={"alias": alias})
            # Try connection with host/port (local instance)
            elif host and port:
                connections.connect(host=host, port=port, alias=alias)
                logger.info("Connected to Milvus using host and port", extra={"alias": alias, "host": host, "port": port})
            # Use environment variables
            else:
                uri_env = os.getenv("MILVUS_HOST")
                token_env = os.getenv("MILVUS_TOKEN")
                port_env = os.getenv("MILVUS_PORT")

                if token_env:
                    connections.connect(uri=uri_env, token=token_env, alias=alias)
                    logger.info("Connected to Milvus using environment URI and token", extra={"alias": alias})
                elif port_env:
                    connections.connect(host=uri_env, port=port_env, alias=alias)
                    logger.info("Connected to Milvus using environment host and port", extra={"alias": alias, "host": uri_env, "port": port_env})
                else:
                    error_msg = "No connection credentials provided"
                    logger.error(error_msg)
                    raise MilvusConnectionError(error_msg)

            if not connections.has_connection(alias):
                error_msg = "Connection to Milvus was not established correctly"
                logger.error(error_msg, extra={"alias": alias})
                raise MilvusConnectionError(error_msg)

        except Exception as e:
            if isinstance(e, MilvusConnectionError):
                raise
            logger.error(
                f"Error creating connection with Milvus: {str(e)}",
                extra={"alias": alias, "error_type": type(e).__name__},
                exc_info=True
            )
            raise MilvusConnectionError(
                f"Error creating connection with Milvus: {str(e)}"
            ) from e

    @staticmethod
    def disconnect(*, alias: str = "default") -> None:
        """
        Closes connection with Milvus.

        Args:
            alias: Connection alias to close.
        """
        logger = get_logger(__name__)
        
        try:
            if connections.has_connection(alias):
                connections.disconnect(alias=alias)
                logger.debug("Disconnected from Milvus", extra={"alias": alias})
        except ConnectionNotExistException:
            # Connection already closed, this is not an error
            logger.debug("Connection already closed", extra={"alias": alias})
            pass
        except Exception as e:
            logger.error(
                f"Error closing connection: {str(e)}",
                extra={"alias": alias, "error_type": type(e).__name__},
                exc_info=True
            )
            raise MilvusConnectionError(
                f"Error closing connection {alias}: {str(e)}"
            ) from e

    @staticmethod
    def load_database(*, dbname: str, alias: str = "default") -> None:
        """
        Loads or creates a database in Milvus.

        Args:
            dbname: Database name.
            alias: Connection alias.

        Raises:
            MilvusConnectionError: If no connection is established.
        """
        logger = get_logger(__name__)
        
        if not connections.has_connection(alias):
            error_msg = f"No connection established with alias '{alias}'"
            logger.error(error_msg)
            raise MilvusConnectionError(error_msg)

        try:
            logger.debug("Loading database", extra={"dbname": dbname, "alias": alias})
            dbs = db.list_database(using=alias)
            if dbname not in dbs:
                db.create_database(dbname, using=alias)
                logger.info("Database created", extra={"dbname": dbname, "alias": alias})
            else:
                logger.debug("Database already exists", extra={"dbname": dbname, "alias": alias})
            db.using_database(db_name=dbname, using=alias)
            logger.info("Database loaded", extra={"dbname": dbname, "alias": alias})
        except Exception as e:
            logger.error(
                f"Error loading database: {str(e)}",
                extra={"dbname": dbname, "alias": alias, "error_type": type(e).__name__},
                exc_info=True
            )
            raise MilvusConnectionError(
                f"Error loading database '{dbname}': {str(e)}"
            ) from e
