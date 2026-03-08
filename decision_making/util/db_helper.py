from decision_making.database.sqlite_helper import SQLiteDB
from decision_making.util.logger import logger

# global variable that will be set in main.py
db = None


def db_initialize(use_local_db: bool = False):
    """Initialize the database connection based on the local-db flag."""
    global db
    if use_local_db:
        _db = SQLiteDB()
        logger.info("SQLite database initialized")
    else:
        pass  # TODO: initialize the connection to other  database here
    db = _db


def get_db():
    """Get the database instance."""
    return db
