# ADR-003: User Authentication & Playlist Architecture

**Status:** Accepted  
**Date:** 2026-04-20  
**Deciders:** Project team  
**Context:** Graduation project — Music Recommendation System

---

## Context

The Music Recommender system needed user-specific features: login/registration, persistent search history, and playlist management. Three key architectural decisions were required.

---

## Decision 1: Database — SQLite (dev) / PostgreSQL (prod)

### Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **SQLite** | Zero setup, single file, dev-friendly | Not suitable for concurrent production workloads |
| **PostgreSQL** | Production-grade, concurrent access, full SQL | Requires server setup |
| **MongoDB** | Flexible schema | Overkill for structured relational data |

### Decision

Use **PostgreSQL** as the primary database with **automatic SQLite fallback** for development when PostgreSQL is unavailable. The connection is detected at startup via `pool_pre_ping`.

```
DATABASE_URL = postgresql://user:pass@host:5432/music_recommender  # prod
DATABASE_URL = sqlite:///music_rec.db                               # fallback
```

**Rationale:** PostgreSQL is the production target for graduation demo. SQLite fallback ensures zero-friction development without needing a running Postgres server during coding.

---

## Decision 2: Authentication — JWT (Stateless)

### Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **Session-based** | Simple server control | Requires session store, not stateless |
| **JWT (chosen)** | Stateless, works with any frontend, no session store | Token revocation requires blocklist |
| **OAuth (Google/Spotify)** | No password management | Complex, requires app approval |

### Decision

Use **JWT (JSON Web Tokens)** via `python-jose`, with **bcrypt password hashing** via `passlib`.

- Tokens expire in 7 days (configurable via `JWT_EXPIRE_MINUTES`)
- Token stored in `localStorage` on the frontend
- FastAPI `Depends()` injection pattern for clean, reusable auth middleware
- Two dependency variants:
  - `get_current_user()` — hard-require auth (401 if missing)
  - `get_optional_user()` — soft-require auth (returns `None` if no token, for /api/search backward compatibility)

**Rationale:** JWT is the industry standard for stateless REST APIs. No session store needed. Works seamlessly with the existing FastAPI architecture.

---

## Decision 3: Playlist — Local System + Spotify Link (Option B)

### Considered Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **Spotify OAuth Playlist Creation** | Native Spotify experience | Requires PKCE flow, limited to 5 test users in Dev Mode, complex |
| **Local Playlist + Spotify Link (chosen)** | No OAuth needed, unlimited users, full control | Playlist only exists in our system |

### Decision

Playlists are managed **entirely within our local database** (PostgreSQL/SQLite). Each track stores:
- `song_name`, `artist_name`, `genre`, `emotion` (from recommendation engine)
- `spotify_url` (captured from `/api/spotify-enrich` enrichment)
- `album_art_url` (captured from `/api/spotify-enrich` enrichment)

The Spotify URL appears as a direct link in the playlist UI — users can click to open the song in Spotify. No Spotify OAuth required.

**Rationale:** Spotify's Development Mode limits apps to 5 approved test users. Creating playlists on behalf of users requires the Authorization Code Flow with PKCE, which is a full OAuth implementation. Option B achieves the same UX goal (listen on Spotify) without these constraints, making it fully functional for a graduation project demo with any number of users.

---

## Schema

```sql
users           (id, username, email, password_hash, created_at)
search_history  (id, user_id FK, query_song, query_artist, model_used, results_count, searched_at)
playlists       (id, user_id FK, name, description, created_at, updated_at)
playlist_tracks (id, playlist_id FK, song_name, artist_name, genre, emotion, spotify_url, album_art_url, position, added_at)
```

---

## Consequences

### Positive
- Zero external service dependencies for auth and playlist storage
- Backward compatible — `/api/search` still works without authentication
- Graceful degradation — app runs even if database is down (returns 503 for auth endpoints)
- SQLite fallback enables immediate development without PostgreSQL

### Negative
- JWT tokens cannot be revoked server-side without a blocklist (acceptable for graduation project)
- Playlists are not synced to Spotify — user must click Spotify link manually
- SQLite fallback has no connection pooling (not suitable for production concurrent load)
