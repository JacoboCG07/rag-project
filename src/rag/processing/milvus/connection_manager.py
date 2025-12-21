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
        try:
            # Try connection with URI (Milvus Cloud)
            if uri and token:
                connections.connect(uri=uri, token=token, alias=alias)
            # Try connection with host/port (local instance)
            elif host and port:
                connections.connect(host=host, port=port, alias=alias)
            # Use environment variables
            else:
                uri_env = os.getenv("MILVUS_HOST")
                token_env = os.getenv("MILVUS_TOKEN")
                port_env = os.getenv("MILVUS_PORT")

                if token_env:
                    connections.connect(uri=uri_env, token=token_env, alias=alias)
                elif port_env:
                    connections.connect(host=uri_env, port=port_env, alias=alias)
                else:
                    raise MilvusConnectionError(
                        "No connection credentials provided"
                    )

            if not connections.has_connection(alias):
                raise MilvusConnectionError(
                    "Connection to Milvus was not established correctly"
                )

        except Exception as e:
            if isinstance(e, MilvusConnectionError):
                raise
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
        try:
            if connections.has_connection(alias):
                connections.disconnect(alias=alias)
        except ConnectionNotExistException:
            # Connection already closed, this is not an error
            pass
        except Exception as e:
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
        if not connections.has_connection(alias):
            raise MilvusConnectionError(
                f"No connection established with alias '{alias}'"
            )

        try:
            dbs = db.list_database(using=alias)
            if dbname not in dbs:
                db.create_database(dbname, using=alias)
            db.using_database(db_name=dbname, using=alias)
        except Exception as e:
            raise MilvusConnectionError(
                f"Error loading database '{dbname}': {str(e)}"
            ) from e
