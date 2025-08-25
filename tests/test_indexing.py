"""Tests for the indexing algorithms."""

import pytest

from app.core.indexing import IndexFactory, KDTreeIndex, LinearSearchIndex, LSHIndex
from app.models.base import Chunk
from app.utils.embedding_utils import create_test_embeddings


class TestLinearSearchIndex:
    """Test Linear Search Index implementation."""

    def test_create_index(self):
        """Test index creation."""
        index = LinearSearchIndex()
        assert not index.is_built
        assert len(index.chunks) == 0
        assert len(index.embeddings) == 0

    def test_add_chunks(self):
        """Test adding chunks to index."""
        index = LinearSearchIndex()

        # Create test chunks
        texts = ["Hello world", "Python programming", "Vector database"]
        embeddings = create_test_embeddings(texts, dimension=10)

        chunks = []
        for text, embedding in zip(texts, embeddings, strict=False):
            chunk = Chunk(text=text, embedding=embedding)
            chunks.append(chunk)

        index.add_chunks(chunks)
        assert len(index.chunks) == 3
        assert len(index.embeddings) == 3

    def test_build_index(self):
        """Test building the index."""
        index = LinearSearchIndex()
        index.build()
        assert index.is_built

    def test_search(self):
        """Test search functionality."""
        index = LinearSearchIndex()

        # Create test chunks
        texts = ["Hello world", "Python programming", "Vector database"]
        embeddings = create_test_embeddings(texts, dimension=10)

        chunks = []
        for text, embedding in zip(texts, embeddings, strict=False):
            chunk = Chunk(text=text, embedding=embedding)
            chunks.append(chunk)

        index.add_chunks(chunks)
        index.build()

        # Search with query
        query_embedding = create_test_embeddings(["Hello"], dimension=10)[0]
        results, scores = index.search(query_embedding, k=2)

        assert len(results) == 2
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)

    def test_search_before_build(self):
        """Test that search fails before building index."""
        index = LinearSearchIndex()

        with pytest.raises(ValueError, match="Index not built"):
            index.search([0.1, 0.2, 0.3], k=1)


class TestKDTreeIndex:
    """Test KD-Tree Index implementation."""

    def test_create_index(self):
        """Test index creation."""
        index = KDTreeIndex()
        assert not index.is_built
        assert index.tree is None
        assert index.dimension is None

    def test_build_index(self):
        """Test building the index."""
        index = KDTreeIndex()

        # Add some chunks
        texts = ["Hello world", "Python programming"]
        embeddings = create_test_embeddings(texts, dimension=3)

        chunks = []
        for text, embedding in zip(texts, embeddings, strict=False):
            chunk = Chunk(text=text, embedding=embedding)
            chunks.append(chunk)

        index.add_chunks(chunks)
        index.build()

        assert index.is_built
        assert index.dimension == 3
        assert index.tree is not None

    def test_search(self):
        """Test search functionality."""
        index = KDTreeIndex()

        # Create test chunks
        texts = ["Hello world", "Python programming", "Vector database"]
        embeddings = create_test_embeddings(texts, dimension=5)

        chunks = []
        for text, embedding in zip(texts, embeddings, strict=False):
            chunk = Chunk(text=text, embedding=embedding)
            chunks.append(chunk)

        index.add_chunks(chunks)
        index.build()

        # Search with query
        query_embedding = create_test_embeddings(["Hello"], dimension=5)[0]
        results, scores = index.search(query_embedding, k=2)

        assert len(results) == 2
        assert len(scores) == 2


class TestLSHIndex:
    """Test LSH Index implementation."""

    def test_create_index(self):
        """Test index creation."""
        index = LSHIndex(num_hashes=5, num_buckets=50)
        assert not index.is_built
        assert index.num_hashes == 5
        assert index.num_buckets == 50

    def test_build_index(self):
        """Test building the index."""
        index = LSHIndex(num_hashes=3, num_buckets=10)

        # Add some chunks
        texts = ["Hello world", "Python programming"]
        embeddings = create_test_embeddings(texts, dimension=4)

        chunks = []
        for text, embedding in zip(texts, embeddings, strict=False):
            chunk = Chunk(text=text, embedding=embedding)
            chunks.append(chunk)

        index.add_chunks(chunks)
        index.build()

        assert index.is_built
        assert len(index.random_planes) == 3
        assert len(index.hash_tables) == 3

    def test_search(self):
        """Test search functionality."""
        index = LSHIndex(num_hashes=2, num_buckets=5)

        # Create test chunks
        texts = ["Hello world", "Python programming", "Vector database"]
        embeddings = create_test_embeddings(texts, dimension=4)

        chunks = []
        for text, embedding in zip(texts, embeddings, strict=False):
            chunk = Chunk(text=text, embedding=embedding)
            chunks.append(chunk)

        index.add_chunks(chunks)
        index.build()

        # Search with query
        query_embedding = create_test_embeddings(["Hello"], dimension=4)[0]
        results, scores = index.search(query_embedding, k=2)

        # LSH might return fewer results due to hashing
        assert len(results) <= 2
        assert len(scores) <= 2


class TestIndexFactory:
    """Test Index Factory."""

    def test_create_linear_index(self):
        """Test creating linear index."""
        index = IndexFactory.create_index("linear")
        assert isinstance(index, LinearSearchIndex)

    def test_create_kdtree_index(self):
        """Test creating KD-tree index."""
        index = IndexFactory.create_index("kdtree")
        assert isinstance(index, KDTreeIndex)

    def test_create_lsh_index(self):
        """Test creating LSH index."""
        index = IndexFactory.create_index("lsh", num_hashes=5, num_buckets=20)
        assert isinstance(index, LSHIndex)
        assert index.num_hashes == 5
        assert index.num_buckets == 20

    def test_create_invalid_index(self):
        """Test creating invalid index type."""
        with pytest.raises(ValueError, match="Unknown index type"):
            IndexFactory.create_index("invalid")

    def test_get_index_info(self):
        """Test getting index information."""
        info = IndexFactory.get_index_info("linear")
        assert "name" in info
        assert "build_time" in info
        assert "search_time" in info
        assert "space" in info
        assert "accuracy" in info
        assert "description" in info

        # Test invalid index type
        invalid_info = IndexFactory.get_index_info("invalid")
        assert "error" in invalid_info
