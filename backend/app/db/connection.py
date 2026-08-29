import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv()

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_SQLITE = "sqlite:///" + os.path.join(_BACKEND_ROOT, "data", "meta_learner.db")
DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_SQLITE)


def _ensure_sqlite_parent(url: str) -> None:
    if not url.startswith("sqlite:///"):
        return
    raw_path = url[len("sqlite:///"):]
    if raw_path in {":memory:", ""} or raw_path.startswith(":memory:"):
        return
    parent = os.path.dirname(raw_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def create_db_engine(database_url: Optional[str] = None):
    """Build an engine. SQLite is the default so synthetic training works without Postgres."""
    url = database_url or DATABASE_URL
    _ensure_sqlite_parent(url)
    if url.startswith("sqlite"):
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency for FastAPI to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(bind=None):
    """Create all tables if they don't exist."""
    from app.db.models import Base
    Base.metadata.create_all(bind=bind or engine)
