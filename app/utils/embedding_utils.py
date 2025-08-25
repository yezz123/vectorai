"""Utility functions for working with embeddings.

Includes Cohere API integration for real embeddings and fallback hash-based ones.
"""

import hashlib
import logging

import cohere
import numpy as np

from app.core.config import get_settings

# Get logger for this module
logger = logging.getLogger(__name__)

# Ensure settings are loaded
settings = get_settings()


class EmbeddingProvider:
    """Provider for creating embeddings using different methods."""

    def __init__(self, cohere_api_key: str | None = None):
        self.cohere_api_key = cohere_api_key or settings.cohere_api_key
        self.cohere_client = None

        if self.cohere_api_key:
            try:
                self.cohere_client = cohere.Client(self.cohere_api_key)
                logger.info("Cohere API client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Cohere client: {e}")
                self.cohere_client = None
        else:
            logger.warning("No Cohere API key provided, using hash-based embeddings")

    def text_to_embedding(self, text: str, dimension: int = 384, use_cohere: bool = True) -> list[float]:
        """Convert text to embedding vector.

        Args:
            text: Input text to convert to embedding
            dimension: Desired embedding dimension (ignored for Cohere)
            use_cohere: Whether to use Cohere API if available

        Returns:
            List of floats representing the embedding vector
        """
        if use_cohere and self.cohere_client:
            try:
                return self._get_cohere_embedding(text)
            except Exception as e:
                logger.warning(f"Cohere API failed, falling back to hash-based embedding: {e}")
                return self._get_hash_embedding(text, dimension)
        else:
            return self._get_hash_embedding(text, dimension)

    def _get_cohere_embedding(self, text: str) -> list[float]:
        """Get embedding from Cohere API."""
        if not self.cohere_client:
            raise ValueError("Cohere client not available")

        response = self.cohere_client.embed(texts=[text], model="embed-english-v3.0", input_type="search_document")

        # Cohere returns embeddings as a list of lists
        embedding = response.embeddings[0]
        return embedding

    def _get_hash_embedding(self, text: str, dimension: int = 384) -> list[float]:
        """Convert text to a simple embedding vector for testing purposes.

        This is a deterministic hash-based approach that generates consistent
        embeddings for the same text. In production, you would use a proper
        embedding model like Cohere, OpenAI, or sentence-transformers.
        """
        # Create a hash of the text
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Convert hash to a list of numbers
        numbers = []
        for i in range(0, len(text_hash), 2):
            if len(numbers) >= dimension:
                break
            hex_pair = text_hash[i : i + 2]
            number = int(hex_pair, 16) / 255.0  # Normalize to [0, 1]
            numbers.append(number)

        # Pad or truncate to desired dimension
        while len(numbers) < dimension:
            numbers.append(0.0)

        # Normalize the vector
        vector = np.array(numbers[:dimension])
        normalized_vector = vector / (np.linalg.norm(vector) + 1e-8)

        return normalized_vector.tolist()

    def batch_text_to_embeddings(
        self, texts: list[str], dimension: int = 384, use_cohere: bool = True
    ) -> list[list[float]]:
        """Convert a batch of texts to embeddings efficiently."""
        if use_cohere and self.cohere_client:
            try:
                return self._get_cohere_batch_embeddings(texts)
            except Exception as e:
                logger.warning(f"Cohere batch API failed, falling back to hash-based embeddings: {e}")
                return [self._get_hash_embedding(text, dimension) for text in texts]
        else:
            return [self._get_hash_embedding(text, dimension) for text in texts]

    def _get_cohere_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get batch embeddings from Cohere API."""
        if not self.cohere_client:
            raise ValueError("Cohere client not available")

        response = self.cohere_client.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")

        return response.embeddings


# Global embedding provider instance
_embedding_provider = None


def get_embedding_provider() -> EmbeddingProvider:
    """Get the global embedding provider instance."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = EmbeddingProvider()
    return _embedding_provider


def text_to_simple_embedding(text: str, dimension: int = 384) -> list[float]:
    """Convert text to a simple embedding vector for testing purposes.

    This is a deterministic hash-based approach that generates consistent
    embeddings for the same text. In production, you would use a proper
    embedding model like Cohere, OpenAI, or sentence-transformers.

    Args:
        text: Input text to convert to embedding
        dimension: Desired embedding dimension (default: 384)

    Returns:
        List of floats representing the embedding vector
    """
    return get_embedding_provider()._get_hash_embedding(text, dimension)


def text_to_cohere_embedding(text: str) -> list[float]:
    """Convert text to embedding using Cohere API.

    Args:
        text: Input text to convert to embedding

    Returns:
        List of floats representing the embedding vector
    """
    return get_embedding_provider().text_to_embedding(text, use_cohere=True)


def create_test_embeddings(texts: list[str], dimension: int = 384, use_cohere: bool = True) -> list[list[float]]:
    """Create test embeddings for a list of texts."""
    return get_embedding_provider().batch_text_to_embeddings(texts, dimension, use_cohere)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")

    vec1_array = np.array(vec1)
    vec2_array = np.array(vec2)

    dot_product = np.dot(vec1_array, vec2_array)
    norm1 = np.linalg.norm(vec1_array)
    norm2 = np.linalg.norm(vec2_array)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")

    vec1_array = np.array(vec1)
    vec2_array = np.array(vec2)

    return np.linalg.norm(vec1_array - vec2_array)


def normalize_vector(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length."""
    vector_array = np.array(vector)
    norm = np.linalg.norm(vector_array)

    if norm == 0:
        return vector

    normalized = vector_array / norm
    return normalized.tolist()


def validate_embedding(embedding: list[float], expected_dimension: int = None) -> bool:
    """Validate that an embedding is properly formatted."""
    if not isinstance(embedding, list):
        return False

    if not all(isinstance(x, int | float) for x in embedding):
        return False

    if expected_dimension and len(embedding) != expected_dimension:
        return False

    return True


def batch_text_to_embeddings(texts: list[str], dimension: int = 384, use_cohere: bool = True) -> list[list[float]]:
    """Convert a batch of texts to embeddings efficiently."""
    return get_embedding_provider().batch_text_to_embeddings(texts, dimension, use_cohere)
