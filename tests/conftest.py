"""
Shared test fixtures for the Music Recommender test suite.

Provides reusable fixtures for config, sample data, and processor instances.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def config():
    """Session-scoped configuration fixture."""
    from hybrid_music_engine.config import get_config
    return get_config()


@pytest.fixture
def sample_song_row():
    """A single song row as pd.Series, simulating a row from song_metadata.csv."""
    return pd.Series({
        'song': 'Shape of You',
        'Artist(s)': 'Ed Sheeran',
        'Genre': 'pop',
        'emotion': 'joy',
        'text': 'The club isn\'t the best place to find a lover',
        'energy': 0.652,
        'danceability': 0.825,
        'valence': 0.931,
        'tempo': 95.977,
        'acousticness': 0.581,
        'instrumentalness': 0.0,
        'speechiness': 0.0802,
        'liveness': 0.0931,
        'key': 1,
        'loudness': -6.749,
        'mode': 0,
    }, name=0)


@pytest.fixture
def sample_audio_df():
    """Small DataFrame with audio features for testing AudioProcessor."""
    return pd.DataFrame({
        'energy': [0.8, 0.3, 0.6],
        'danceability': [0.7, 0.4, 0.5],
        'valence': [0.9, 0.2, 0.5],
        'tempo': [120.0, 85.0, 100.0],
        'acousticness': [0.1, 0.7, 0.4],
        'instrumentalness': [0.0, 0.3, 0.1],
        'speechiness': [0.05, 0.08, 0.06],
        'liveness': [0.1, 0.15, 0.12],
        'key': [0, 5, 3],
        'loudness': [-5.0, -10.0, -7.0],
        'mode': [1, 0, 1],
    })


@pytest.fixture
def sample_song_df():
    """DataFrame simulating song_metadata.csv for engine tests."""
    return pd.DataFrame({
        'song': ['Shape of You', 'Someone Like You', 'Blinding Lights'],
        'Artist(s)': ['Ed Sheeran', 'Adele', 'The Weeknd'],
        'Genre': ['pop', 'soul', 'synth-pop'],
        'emotion': ['joy', 'sadness', 'joy'],
        'energy': [0.652, 0.264, 0.730],
        'danceability': [0.825, 0.479, 0.514],
        'valence': [0.931, 0.140, 0.334],
        'tempo': [95.977, 67.09, 171.005],
        'acousticness': [0.581, 0.572, 0.00127],
        'instrumentalness': [0.0, 0.0, 0.0000954],
        'speechiness': [0.0802, 0.0316, 0.0598],
        'liveness': [0.0931, 0.101, 0.0897],
        'key': [1, 6, 1],
        'loudness': [-6.749, -8.359, -5.934],
        'mode': [0, 1, 1],
    })
