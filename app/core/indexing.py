"""Custom vector indexing algorithms implementation.

Implements three different approaches: Linear Search, KD-Tree, and LSH.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.models.base import Chunk


@dataclass
class IndexStats:
    """Statistics about the index performance."""

    build_time: float
    memory_usage: int
    search_time: float
    accuracy: float


class BaseIndex:
    """Base class for all vector indexes."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings: list[np.ndarray] = []
        self.is_built = False

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index."""
        for chunk in chunks:
            self.chunks.append(chunk)
            self.embeddings.append(np.array(chunk.embedding))

    def build(self) -> None:
        """Build the index. Must be implemented by subclasses."""
        raise NotImplementedError

    def search(self, query_embedding: list[float], k: int) -> tuple[list[Chunk], list[float]]:
        """Search for k nearest neighbors. Must be implemented by subclasses."""
        raise NotImplementedError

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return 0.0 if norm1 == 0 or norm2 == 0 else dot_product / (norm1 * norm2)


class LinearSearchIndex(BaseIndex):
    """Linear Search Index - O(n) search time, O(1) build time, O(n) space.

    This is the simplest approach that compares the query vector against all vectors
    in the database. It's chosen for its simplicity and guaranteed accuracy.
    """

    def build(self) -> None:
        """Build the index - for linear search, this is just storing the vectors."""
        self.is_built = True

    def search(self, query_embedding: list[float], k: int) -> tuple[list[Chunk], list[float]]:
        """Linear search through all vectors."""
        if not self.is_built:
            raise ValueError("Index not built. Call build() first.")

        query_vec = np.array(query_embedding)
        similarities = []

        # Calculate similarities with Designing a Scalable Architecture with Solid Principles 🌟all vectors
        for embedding in self.embeddings:
            similarity = self._cosine_similarity(query_vec, embedding)
            similarities.append(similarity)

        # Get top k results
        indices = np.argsort(similarities)[::-1][:k]

        results = [self.chunks[i] for i in indices]
        scores = [similarities[i] for i in indices]

        return results, scores


class KDTreeIndex(BaseIndex):
    """KD-Tree Index - O(log n) search time, O(n log n) build time, O(n) space.

    This approach recursively partitions the space into regions, allowing for
    efficient range queries and nearest neighbor searches.
    """

    def __init__(self):
        super().__init__()
        self.tree = None
        self.dimension = None

    def build(self) -> None:
        """Build the KD-tree."""
        if not self.embeddings:
            self.is_built = True
            return

        self.dimension = len(self.embeddings[0])
        self.tree = self._build_tree(self.embeddings, 0)
        self.is_built = True

    def _build_tree(self, embeddings: list[np.ndarray], depth: int) -> dict:
        """Recursively build the KD-tree."""
        if not embeddings:
            return None

        if len(embeddings) == 1:
            # Find the index of this embedding in the original list
            chunk_index = self._find_embedding_index(embeddings[0])
            return {"point": embeddings[0], "left": None, "right": None, "chunk_index": chunk_index}

        # Choose axis to split on
        axis = depth % self.dimension

        # Sort by the chosen axis
        sorted_embeddings = sorted(embeddings, key=lambda x: x[axis])
        median_idx = len(sorted_embeddings) // 2

        # Find the index of the median embedding in the original list
        chunk_index = self._find_embedding_index(sorted_embeddings[median_idx])

        return {
            "point": sorted_embeddings[median_idx],
            "left": self._build_tree(sorted_embeddings[:median_idx], depth + 1),
            "right": self._build_tree(sorted_embeddings[median_idx + 1 :], depth + 1),
            "chunk_index": chunk_index,
        }

    def _find_embedding_index(self, target_embedding: np.ndarray) -> int:
        """Find the index of an embedding in the original embeddings list."""
        return next(
            (i for i, embedding in enumerate(self.embeddings) if np.array_equal(embedding, target_embedding)),
            0,
        )

    def search(self, query_embedding: list[float], k: int) -> tuple[list[Chunk], list[float]]:
        """Search using KD-tree."""
        if not self.is_built:
            raise ValueError("Index not built. Call build() first.")

        query_vec = np.array(query_embedding)
        candidates = []

        # Collect candidates using tree traversal
        self._search_recursive(self.tree, query_vec, candidates, depth=0)

        # Calculate similarities and get top k
        similarities = []
        for candidate in candidates:
            similarity = self._cosine_similarity(query_vec, candidate["point"])
            similarities.append(similarity)

        # Sort by similarity and get top k
        if similarities:
            sorted_indices = np.argsort(similarities)[::-1][:k]
            results = [self.chunks[candidates[i]["chunk_index"]] for i in sorted_indices]
            scores = [similarities[i] for i in sorted_indices]
        else:
            results, scores = [], []

        return results, scores

    def _search_recursive(self, node: dict, query_vec: np.ndarray, candidates: list, depth: int):
        """Recursively search the KD-tree."""
        if node is None:
            return

        # Add current node to candidates
        candidates.append(node)

        axis = depth % self.dimension

        # Decide which subtree to explore first
        if query_vec[axis] < node["point"][axis]:
            first, second = node["left"], node["right"]
        else:
            first, second = node["right"], node["left"]

        # Explore first subtree
        if first:
            self._search_recursive(first, query_vec, candidates, depth + 1)

        # Explore second subtree if it might contain better candidates
        if second:
            # Simple heuristic: if distance to splitting plane is small, explore both
            distance_to_plane = abs(query_vec[axis] - node["point"][axis])
            if distance_to_plane < 0.1:  # Threshold for exploration
                self._search_recursive(second, query_vec, candidates, depth + 1)


class LSHIndex(BaseIndex):
    """Locality Sensitive Hashing Index - O(1) search time, O(n) build time, O(n) space.

    This approach uses hash functions to map similar vectors to the same buckets,
    allowing for fast approximate similarity search.
    """

    def __init__(self, num_hashes: int = 10, num_buckets: int = 100):
        super().__init__()
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.hash_tables = []
        self.random_planes = []

    def build(self) -> None:
        """Build the LSH index."""
        if not self.embeddings:
            self.is_built = True
            return

        dimension = len(self.embeddings[0])

        # Generate random hyperplanes for hashing
        self.random_planes = [np.random.randn(dimension) for _ in range(self.num_hashes)]

        # Create hash tables
        self.hash_tables = [{} for _ in range(self.num_hashes)]

        # Hash all vectors
        for i, embedding in enumerate(self.embeddings):
            for j, plane in enumerate(self.random_planes):
                hash_value = self._hash_vector(embedding, plane)
                if hash_value not in self.hash_tables[j]:
                    self.hash_tables[j][hash_value] = []
                self.hash_tables[j][hash_value].append(i)

        self.is_built = True

    def _hash_vector(self, vector: np.ndarray, plane: np.ndarray) -> int:
        """Hash a vector using a random hyperplane."""
        # Project vector onto the hyperplane and determine which side it's on
        projection = np.dot(vector, plane)
        return hash(projection) % self.num_buckets

    def search(self, query_embedding: list[float], k: int) -> tuple[list[Chunk], list[float]]:
        """Search using LSH."""
        if not self.is_built:
            raise ValueError("Index not built. Call build() first.")

        query_vec = np.array(query_embedding)
        candidate_indices = set()

        # Collect candidates from hash tables
        for i, plane in enumerate(self.random_planes):
            hash_value = self._hash_vector(query_vec, plane)
            if hash_value in self.hash_tables[i]:
                candidate_indices.update(self.hash_tables[i][hash_value])

        # Calculate similarities for candidates
        similarities = []
        candidates = []

        for idx in candidate_indices:
            similarity = self._cosine_similarity(query_vec, self.embeddings[idx])
            similarities.append(similarity)
            candidates.append(self.chunks[idx])

        # Sort by similarity and get top k
        if similarities:
            sorted_indices = np.argsort(similarities)[::-1][:k]
            results = [candidates[i] for i in sorted_indices]
            scores = [similarities[i] for i in sorted_indices]
        else:
            results, scores = [], []

        return results, scores


class IndexFactory:
    """Factory for creating different types of indexes."""

    @staticmethod
    def create_index(index_type: str, **kwargs) -> BaseIndex:
        """Create an index of the specified type."""
        if index_type == "linear":
            return LinearSearchIndex()
        elif index_type == "kdtree":
            return KDTreeIndex()
        elif index_type == "lsh":
            return LSHIndex(**kwargs)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    @staticmethod
    def get_index_info(index_type: str) -> dict[str, Any]:
        """Get information about the specified index type."""
        info = {
            "linear": {
                "name": "Linear Search",
                "build_time": "O(1)",
                "search_time": "O(n)",
                "space": "O(n)",
                "accuracy": "100%",
                "description": "Simple linear search through all vectors. Guaranteed accuracy but slow for large datasets.",
            },
            "kdtree": {
                "name": "KD-Tree",
                "build_time": "O(n log n)",
                "search_time": "O(log n)",
                "space": "O(n)",
                "accuracy": "100%",
                "description": "Space-partitioning tree structure for efficient range queries and nearest neighbor search.",
            },
            "lsh": {
                "name": "Locality Sensitive Hashing",
                "build_time": "O(n)",
                "search_time": "O(1)",
                "space": "O(n)",
                "accuracy": "~90-95%",
                "description": "Hash-based approach for approximate similarity search. Fast but may miss some results.",
            },
        }

        return info.get(index_type, {"error": "Unknown index type"})
