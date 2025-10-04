"""SQLite session placeholder.

If you later adopt SQLAlchemy, define engine/sessionmakers here.
For simple use-cases, the standard library `sqlite3` may be sufficient.
"""

import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("SQLITE_PATH", "rag_app.db")).resolve()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn
