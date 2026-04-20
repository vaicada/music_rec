"""
Unit tests for hybrid_music_engine.inference module.

Tests FAISSIndex (search, column detection) and MusicRecommendationEngine
(_find_song, _get_column_names).
"""

import pytest
import numpy as np
import pandas as pd

from hybrid_music_engine.inference import FAISSIndex, MusicRecommendationEngine


# =============================================================================
# FAISSIndex Tests
# =============================================================================

class TestFAISSIndex:
    """Tests for FAISSIndex without requiring a loaded index."""

    @pytest.fixture(autouse=True)
    def setup(self, config):
        self.index = FAISSIndex(config)

    @pytest.mark.unit
    def test_search_raises_without_index(self):
        """search() should raise ValueError when index is not loaded."""
        with pytest.raises(ValueError, match="Index not loaded"):
            self.index.search(np.zeros(64), top_k=5)

    @pytest.mark.unit
    def test_reconstruct_raises_without_index(self):
        """reconstruct() should raise ValueError when index is not loaded."""
        with pytest.raises(ValueError, match="Index not loaded"):
            self.index.reconstruct(0)

    @pytest.mark.unit
    def test_initial_state(self):
        """New FAISSIndex should have empty state."""
        assert self.index.index is None
        assert self.index.song_ids == []
        assert self.index.idx_to_song == {}
        assert self.index.using_gpu is False

    @pytest.mark.unit
    def test_get_column_names_standard(self):
        """_get_column_names should detect 'song' and 'Artist(s)' columns."""
        df = pd.DataFrame({'song': [], 'Artist(s)': [], 'Genre': []})
        song_col, artist_col = self.index._get_column_names(df)
        assert song_col == 'song'
        assert artist_col == 'Artist(s)'

    @pytest.mark.unit
    def test_get_column_names_alternative(self):
        """_get_column_names should detect 'name' and 'artist' columns."""
        df = pd.DataFrame({'name': [], 'artist': [], 'genre': []})
        song_col, artist_col = self.index._get_column_names(df)
        assert song_col == 'name'
        assert artist_col == 'artist'

    @pytest.mark.unit
    def test_get_column_names_tracks_format(self):
        """_get_column_names should detect 'song_name' and 'artists' columns."""
        df = pd.DataFrame({'song_name': [], 'artists': [], 'genre': []})
        song_col, artist_col = self.index._get_column_names(df)
        assert song_col == 'song_name'
        assert artist_col == 'artists'


# =============================================================================
# MusicRecommendationEngine Tests
# =============================================================================

class TestMusicRecommendationEngine:
    """Tests for MusicRecommendationEngine helper methods."""

    @pytest.fixture(autouse=True)
    def setup(self, config):
        self.engine = MusicRecommendationEngine(config)

    @pytest.mark.unit
    def test_find_song_returns_none_without_data(self):
        """_find_song should return None when song_data is not loaded."""
        result = self.engine._find_song("Shape of You")
        assert result is None

    @pytest.mark.unit
    def test_find_song_exact_match(self, sample_song_df):
        """_find_song should find exact song name match (case-insensitive)."""
        self.engine.song_data = sample_song_df
        result = self.engine._find_song("shape of you")
        assert result is not None
        assert result['song'] == 'Shape of You'

    @pytest.mark.unit
    def test_find_song_with_artist(self, sample_song_df):
        """_find_song should filter by artist when provided."""
        self.engine.song_data = sample_song_df
        result = self.engine._find_song("shape of you", "ed sheeran")
        assert result is not None
        assert result['Artist(s)'] == 'Ed Sheeran'

    @pytest.mark.unit
    def test_find_song_fuzzy_match(self, sample_song_df):
        """_find_song should fallback to fuzzy match (contains)."""
        self.engine.song_data = sample_song_df
        result = self.engine._find_song("shape")
        assert result is not None
        assert "Shape" in result['song']

    @pytest.mark.unit
    def test_find_song_not_found(self, sample_song_df):
        """_find_song should return None for non-existent songs."""
        self.engine.song_data = sample_song_df
        result = self.engine._find_song("zzz_nonexistent_song_zzz")
        assert result is None

    @pytest.mark.unit
    def test_get_column_names_from_engine(self, sample_song_df):
        """_get_column_names should work on engine's song_data."""
        self.engine.song_data = sample_song_df
        song_col, artist_col = self.engine._get_column_names()
        assert song_col == 'song'
        assert artist_col == 'Artist(s)'

    @pytest.mark.unit
    def test_recommendations_by_mood_without_data(self):
        """get_recommendations_by_mood should return [] without song_data."""
        result = self.engine.get_recommendations_by_mood("happy")
        assert result == []

    @pytest.mark.unit
    def test_recommendations_by_mood_happy(self, sample_song_df):
        """get_recommendations_by_mood('happy') should return joy songs."""
        self.engine.song_data = sample_song_df
        results = self.engine.get_recommendations_by_mood("happy", top_k=5)
        assert isinstance(results, list)
        # Results should contain songs with 'joy' emotion
        for r in results:
            assert 'song' in r
            assert 'artist' in r
