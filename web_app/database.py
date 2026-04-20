"""
Database Models & Engine for Music Recommender.

Supports PostgreSQL (primary) with SQLite fallback for development.
All tables auto-create on first import via `create_all()`.

Models:
    - User: authentication accounts
    - SearchHistory: logged search queries per user
    - Playlist: user-created playlists
    - PlaylistTrack: individual tracks within playlists

Usage:
    from database import get_db, User, Playlist, PlaylistTrack, SearchHistory
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

# =============================================================================
# Database Connection
# =============================================================================

# Priority: DATABASE_URL env var → PostgreSQL → SQLite fallback
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    # Try PostgreSQL with common defaults
    PG_USER = os.getenv("PG_USER", "postgres")
    PG_PASS = os.getenv("PG_PASS", "postgres")
    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = os.getenv("PG_PORT", "5432")
    PG_DB = os.getenv("PG_DB", "music_recommender")
    DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# Try connecting to PostgreSQL; fall back to SQLite on failure
try:
    _test_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    with _test_engine.connect() as conn:
        conn.execute(conn.connection.cursor().execute("SELECT 1") if False else __import__("sqlalchemy").text("SELECT 1"))
    engine = _test_engine
    DB_TYPE = "postgresql"
    print(f"[DB] Connected to PostgreSQL: {PG_HOST}:{PG_PORT}/{PG_DB}" if "postgresql" in DATABASE_URL else f"[DB] Connected: {DATABASE_URL[:50]}...")
except Exception as pg_err:
    # Fallback to SQLite
    db_dir = os.path.dirname(os.path.abspath(__file__))
    sqlite_path = os.path.join(db_dir, "music_rec.db")
    DATABASE_URL = f"sqlite:///{sqlite_path}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    DB_TYPE = "sqlite"
    print(f"[DB] PostgreSQL unavailable ({pg_err}), using SQLite: {sqlite_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency: yields a database session, auto-closes on finish."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# ORM Base
# =============================================================================

class Base(DeclarativeBase):
    pass


# =============================================================================
# Models
# =============================================================================

class User(Base):
    """User account for authentication."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    search_history = relationship("SearchHistory", back_populates="user", cascade="all, delete-orphan")
    playlists = relationship("Playlist", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"


class SearchHistory(Base):
    """Record of a user's search query and results."""

    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    query_song = Column(String(255), nullable=False)
    query_artist = Column(String(255), default="")
    model_used = Column(String(20), default="model1")
    results_count = Column(Integer, default=0)
    searched_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="search_history")

    def __repr__(self) -> str:
        return f"<SearchHistory(id={self.id}, song='{self.query_song}')>"


class Playlist(Base):
    """User-created playlist."""

    __tablename__ = "playlists"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="playlists")
    tracks = relationship("PlaylistTrack", back_populates="playlist", cascade="all, delete-orphan",
                          order_by="PlaylistTrack.position")

    def __repr__(self) -> str:
        return f"<Playlist(id={self.id}, name='{self.name}')>"


class PlaylistTrack(Base):
    """Individual track within a playlist, linked to Spotify."""

    __tablename__ = "playlist_tracks"

    id = Column(Integer, primary_key=True, index=True)
    playlist_id = Column(Integer, ForeignKey("playlists.id", ondelete="CASCADE"), nullable=False)
    song_name = Column(String(255), nullable=False)
    artist_name = Column(String(255), default="")
    genre = Column(String(100), default="")
    emotion = Column(String(100), default="")
    spotify_url = Column(String(500), default="")
    album_art_url = Column(String(500), default="")
    position = Column(Integer, default=0)
    added_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    playlist = relationship("Playlist", back_populates="tracks")

    def __repr__(self) -> str:
        return f"<PlaylistTrack(id={self.id}, song='{self.song_name}')>"


# =============================================================================
# Create Tables
# =============================================================================

Base.metadata.create_all(bind=engine)
print(f"[DB] All tables ready ({DB_TYPE}): users, search_history, playlists, playlist_tracks")
