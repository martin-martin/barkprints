"""Tests for embedding matching."""

import numpy as np

from barkprints.corpus import Corpus
from barkprints.embedding_matcher import EmbeddingMatcher


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    matcher = EmbeddingMatcher()
    
    # Identical vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    assert np.isclose(matcher.cosine_similarity(v1, v2), 1.0)
    
    # Opposite vectors
    v3 = np.array([1, 0, 0])
    v4 = np.array([-1, 0, 0])
    assert np.isclose(matcher.cosine_similarity(v3, v4), -1.0)
    
    # Orthogonal vectors
    v5 = np.array([1, 0, 0])
    v6 = np.array([0, 1, 0])
    assert np.isclose(matcher.cosine_similarity(v5, v6), 0.0)


def test_compute_similarities():
    """Test computing similarities with multiple embeddings."""
    matcher = EmbeddingMatcher()
    
    image_emb = np.array([1.0, 0.0, 0.0])
    corpus_embs = np.array([
        [1.0, 0.0, 0.0],  # Same as image
        [0.0, 1.0, 0.0],  # Orthogonal
        [-1.0, 0.0, 0.0],  # Opposite
    ])
    
    sims = matcher.compute_similarities(image_emb, corpus_embs)
    
    assert len(sims) == 3
    assert sims[0] > sims[1]  # Same direction is more similar
    assert sims[2] < sims[1]  # Opposite is least similar


def test_find_nearest_single():
    """Test finding single nearest match."""
    sentences = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    corpus = Corpus("test", sentences, embeddings)
    
    matcher = EmbeddingMatcher()
    image_emb = np.array([0.9, 0.1, 0.0])  # Closest to first
    
    results = matcher.find_nearest(image_emb, corpus, top_k=1)
    
    assert len(results) == 1
    assert results[0][0] == "First sentence."
    assert 0 <= results[0][1] <= 1  # Score in valid range


def test_find_nearest_top_k():
    """Test finding top-k matches."""
    sentences = [f"Sentence {i}." for i in range(5)]
    embeddings = np.random.randn(5, 10)
    corpus = Corpus("test", sentences, embeddings)
    
    matcher = EmbeddingMatcher()
    image_emb = np.random.randn(10)
    
    results = matcher.find_nearest(image_emb, corpus, top_k=3)
    
    assert len(results) == 3
    # Check scores are in descending order
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_find_nearest_deterministic():
    """Test that matching is deterministic."""
    sentences = ["First.", "Second.", "Third."]
    embeddings = np.random.randn(3, 10)
    corpus = Corpus("test", sentences, embeddings)
    
    matcher = EmbeddingMatcher()
    image_emb = np.random.randn(10)
    
    results1 = matcher.find_nearest(image_emb, corpus, top_k=2)
    results2 = matcher.find_nearest(image_emb, corpus, top_k=2)
    
    assert results1 == results2

