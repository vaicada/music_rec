"""
Unit tests for hybrid_music_engine.processors module.

Tests TextProcessor (lyrics cleaning, tokenization) and AudioProcessor
(normalization, feature extraction).
"""

import pytest
import torch
import numpy as np
import pandas as pd

from hybrid_music_engine.processors import TextProcessor, AudioProcessor


# =============================================================================
# TextProcessor Tests
# =============================================================================

class TestTextProcessor:
    """Tests for TextProcessor.clean_lyrics and combine_text_features."""

    @pytest.fixture(autouse=True)
    def setup(self, config):
        self.processor = TextProcessor(config)

    # --- clean_lyrics ---

    @pytest.mark.unit
    def test_clean_lyrics_removes_section_markers(self):
        result = self.processor.clean_lyrics("[Verse 1] Hello world [Chorus] Yeah")
        assert "[Verse 1]" not in result
        assert "[Chorus]" not in result
        assert "hello world" in result

    @pytest.mark.unit
    def test_clean_lyrics_handles_none(self):
        assert self.processor.clean_lyrics(None) == ""

    @pytest.mark.unit
    def test_clean_lyrics_handles_nan(self):
        assert self.processor.clean_lyrics(float('nan')) == ""

    @pytest.mark.unit
    def test_clean_lyrics_handles_empty_string(self):
        assert self.processor.clean_lyrics("") == ""

    @pytest.mark.unit
    def test_clean_lyrics_removes_urls(self):
        result = self.processor.clean_lyrics("Check out https://example.com for more")
        assert "https://" not in result
        assert "example.com" not in result

    @pytest.mark.unit
    def test_clean_lyrics_removes_html_tags(self):
        result = self.processor.clean_lyrics("<b>Bold</b> and <i>italic</i>")
        assert "<b>" not in result
        assert "<i>" not in result
        assert "bold" in result

    @pytest.mark.unit
    def test_clean_lyrics_lowercases_text(self):
        result = self.processor.clean_lyrics("HELLO WORLD")
        assert result == "hello world"

    @pytest.mark.unit
    def test_clean_lyrics_normalizes_whitespace(self):
        result = self.processor.clean_lyrics("hello    world   test")
        assert "    " not in result
        assert "hello world test" == result

    # --- combine_text_features ---

    @pytest.mark.unit
    def test_combine_text_features_with_all_fields(self):
        result = self.processor.combine_text_features("lyrics text", "joy", "pop")
        assert "[EMOTION: joy]" in result
        assert "[GENRE: pop]" in result
        assert "lyrics text" in result

    @pytest.mark.unit
    def test_combine_text_features_with_only_lyrics(self):
        result = self.processor.combine_text_features("lyrics text", "", "")
        assert result == "lyrics text"
        assert "[EMOTION" not in result

    @pytest.mark.unit
    def test_combine_text_features_with_empty_lyrics(self):
        result = self.processor.combine_text_features("", "joy", "pop")
        assert "[EMOTION: joy]" in result
        assert "[GENRE: pop]" in result


# =============================================================================
# AudioProcessor Tests
# =============================================================================

class TestAudioProcessor:
    """Tests for AudioProcessor.fit and transform."""

    @pytest.fixture(autouse=True)
    def setup(self, config):
        self.processor = AudioProcessor(config)

    @pytest.mark.unit
    def test_fit_returns_self(self, sample_audio_df):
        """fit() should return self for method chaining."""
        result = self.processor.fit(sample_audio_df)
        assert result is self.processor

    @pytest.mark.unit
    def test_fit_computes_statistics(self, sample_audio_df):
        """After fit(), internal normalization stats should be populated."""
        self.processor.fit(sample_audio_df)
        assert self.processor._mean is not None
        assert self.processor._std is not None

    @pytest.mark.unit
    def test_transform_output_is_tensor(self, sample_audio_df):
        """transform() should return a torch.Tensor."""
        self.processor.fit(sample_audio_df)
        result = self.processor.transform(sample_audio_df)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.unit
    def test_transform_output_shape(self, sample_audio_df):
        """transform() output shape should be (n_samples, n_features)."""
        self.processor.fit(sample_audio_df)
        result = self.processor.transform(sample_audio_df)
        n_features = len(self.processor.feature_names)
        assert result.shape == (len(sample_audio_df), n_features)

    @pytest.mark.unit
    def test_transform_output_dtype(self, sample_audio_df):
        """transform() should return float32 tensors."""
        self.processor.fit(sample_audio_df)
        result = self.processor.transform(sample_audio_df)
        assert result.dtype == torch.float32

    @pytest.mark.unit
    def test_feature_names_from_config(self, config):
        """feature_names should come from config.audio.audio_features."""
        assert self.processor.feature_names == config.audio.audio_features
