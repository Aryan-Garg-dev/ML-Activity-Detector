"""Query-level response cache with TTL expiry, normalized key hashing,
and sentence-embedding semantic similarity matching.

Provides in-memory caching of AgentResponse objects to avoid recomputing
results for repeated or semantically-identical queries. Three lookup modes
in priority order:

1. Exact hash (always active): SHA-256 of normalized query — O(1), zero cost.
2. Sentence-embedding similarity (primary semantic): fastembed ONNX model computes
   dense query vectors and uses cosine similarity. Catches paraphrases like
   "suspicious activity in records" ≈ "anomalous transactions" that TF-IDF misses.
3. TF-IDF similarity (fallback): Used only when fastembed is unavailable. Keeps
   the cache useful even without the embedding model downloaded.
"""

import hashlib
import re
import time
from typing import Any
from loguru import logger
from schemas.contracts import AgentResponse

# --- Embedding backends — tried in priority order ---

# Priority 1: fastembed (ONNX, local, no PyTorch dependency)
_FASTEMBED_AVAILABLE = False
try:
    from fastembed import TextEmbedding  # type: ignore
    _FASTEMBED_AVAILABLE = True
except ImportError:
    pass

# Priority 2: scikit-learn TF-IDF (lightweight fallback)
_SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

# Lazy-loaded fastembed model singleton — downloaded on first cache semantic lookup
_embed_model: Any = None
# Default model: small (33MB), fast, excellent for short English sentences
_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# AML query stopwords — filler words that should not affect cache key uniqueness.
# "Is there any suspicious activity in the records" == "Is there any suspicious activity"
_QUERY_STOPWORDS = frozenset({
    "the", "a", "an", "in", "at", "on", "of", "for", "to", "any",
    "please", "can", "you", "me", "my", "our", "show", "find", "get",
    "list", "all", "some", "records", "database", "data", "dataset",
    "report", "results", "transactions", "accounts", "activity",
    "there", "is", "are", "was", "were", "do", "does", "did",
    "have", "has", "had", "be", "been", "being",
})


def _get_embed_model() -> Any:
    """Lazy-initialize the fastembed model singleton.

    Downloads model on first call (~33MB for bge-small-en-v1.5), then caches in memory.
    Thread-safe for read-only inference after initialization.
    """
    global _embed_model
    if _embed_model is None and _FASTEMBED_AVAILABLE:
        try:
            logger.info(
                "Initializing fastembed model '{model}' for semantic cache similarity",
                model=_EMBED_MODEL_NAME,
            )
            _embed_model = TextEmbedding(model_name=_EMBED_MODEL_NAME)
            logger.info("fastembed model loaded — semantic cache is fully active")
        except Exception as e:
            logger.warning(
                "fastembed model initialization failed ({err}). "
                "Falling back to TF-IDF semantic similarity.",
                err=e,
            )
    return _embed_model


def _cosine_similarity_numpy(v1: Any, v2: Any) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""
    import numpy as np
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return dot / norm if norm > 0.0 else 0.0


class CacheEntry:
    """Single cache entry wrapping an AgentResponse with timestamp and query text."""

    __slots__ = ("response", "created_at", "hit_count", "normalized_query", "embedding")

    def __init__(self, response: AgentResponse, normalized_query: str, embedding: Any = None) -> None:
        self.response = response
        self.created_at = time.monotonic()
        self.hit_count = 0
        self.normalized_query = normalized_query
        # Pre-computed embedding vector (numpy array or None if model unavailable)
        self.embedding = embedding


class QueryCache:
    """In-memory TTL-based cache for AgentResponse results.

    Lookup priority:
    1. Exact SHA-256 hash match (always, O(1))
    2. Sentence-embedding cosine similarity via fastembed (primary semantic)
    3. TF-IDF cosine similarity via scikit-learn (semantic fallback)

    Usage:
        cache = QueryCache(ttl_minutes=30, semantic_enabled=True)
        result = cache.get("Find structuring patterns")
        if result is None:
            result = run_agent(...)
            cache.put("Find structuring patterns", result)
    """

    def __init__(
        self,
        ttl_minutes: int = 30,
        enabled: bool = True,
        semantic_enabled: bool = True,
        semantic_threshold: float = 0.78,
    ) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._ttl_seconds = ttl_minutes * 60
        self._enabled = enabled
        self._semantic_enabled = semantic_enabled
        self._semantic_threshold = semantic_threshold
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._embed_hits = 0
        self._tfidf_hits = 0

        if semantic_enabled and not _FASTEMBED_AVAILABLE and not _SKLEARN_AVAILABLE:
            logger.warning(
                "Semantic cache requested but neither fastembed nor scikit-learn is installed. "
                "Falling back to exact hash matching only."
            )
        elif semantic_enabled and _FASTEMBED_AVAILABLE:
            logger.debug(
                "Semantic cache configured: fastembed (primary) + TF-IDF (fallback), threshold={thr}",
                thr=semantic_threshold,
            )
        elif semantic_enabled and _SKLEARN_AVAILABLE:
            logger.debug(
                "Semantic cache configured: TF-IDF only (fastembed unavailable), threshold={thr}",
                thr=semantic_threshold,
            )

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query for consistent cache key and embedding input.

        Steps:
        1. Lowercase and strip whitespace
        2. Collapse multiple spaces
        3. Remove trailing punctuation
        4. Strip AML-domain stopwords that do not affect intent
           (e.g. "in the records", "please show me")
        """
        normalized = query.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.rstrip("?.!")

        # Remove stopwords — keeping only intent-bearing tokens for embedding
        tokens = normalized.split()
        tokens = [t for t in tokens if t not in _QUERY_STOPWORDS]
        return " ".join(tokens) if tokens else normalized

    @staticmethod
    def _hash_key(normalized_query: str) -> str:
        """Generate a stable SHA-256 hash key from a normalized query string."""
        return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]

    def _embed_query(self, text: str) -> Any:
        """Compute sentence embedding for the given text using fastembed.

        Returns a numpy array, or None if the model is unavailable.
        """
        model = _get_embed_model()
        if model is None:
            return None
        try:
            import numpy as np
            # fastembed.embed() returns a generator of numpy arrays
            vectors = list(model.embed([text]))
            return np.array(vectors[0])
        except Exception as e:
            logger.debug("fastembed inference failed ({err}), skipping embedding lookup", err=e)
            return None

    def _find_embedding_match(self, query_vec: Any) -> str | None:
        """Find the cache entry with highest cosine similarity using pre-stored embeddings.

        Returns the cache key of the best match above threshold, or None.
        """
        import numpy as np
        best_key: str | None = None
        best_sim = -1.0
        now = time.monotonic()

        for key, entry in self._store.items():
            if now - entry.created_at > self._ttl_seconds:
                continue
            if entry.embedding is None:
                continue
            sim = _cosine_similarity_numpy(query_vec, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is not None and best_sim >= self._semantic_threshold:
            logger.debug(
                "Embedding cache match: similarity={sim:.3f} (threshold={thr}) key={key}",
                sim=best_sim,
                thr=self._semantic_threshold,
                key=best_key,
            )
            return best_key

        return None

    def _find_tfidf_match(self, normalized_query: str) -> str | None:
        """TF-IDF cosine similarity fallback when fastembed is unavailable.

        Returns the cache key of the best match above threshold, or None.
        """
        if not _SKLEARN_AVAILABLE or not self._store:
            return None

        now = time.monotonic()
        live_keys: list[str] = []
        live_queries: list[str] = []

        for key, entry in self._store.items():
            if now - entry.created_at <= self._ttl_seconds:
                live_keys.append(key)
                live_queries.append(entry.normalized_query)

        if not live_queries:
            return None

        try:
            all_queries = live_queries + [normalized_query]
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            tfidf_matrix = vectorizer.fit_transform(all_queries)
            target_vec = tfidf_matrix[-1]
            stored_vecs = tfidf_matrix[:-1]
            similarities = sklearn_cosine(target_vec, stored_vecs).flatten()
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self._semantic_threshold:
                logger.debug(
                    "TF-IDF cache match: similarity={sim:.3f} key={key}",
                    sim=best_sim,
                    key=live_keys[best_idx],
                )
                return live_keys[best_idx]
        except Exception as e:
            logger.debug("TF-IDF similarity check failed ({err}), skipping", err=e)

        return None

    def get(self, query: str) -> AgentResponse | None:
        """Retrieve cached response for a query, or None if cache miss or expired.

        Lookup order:
        1. Exact hash (O(1))
        2. Sentence-embedding cosine similarity (fastembed, primary semantic)
        3. TF-IDF cosine similarity (sklearn, fallback when fastembed unavailable)
        """
        if not self._enabled:
            return None

        normalized = self._normalize_query(query)
        key = self._hash_key(normalized)

        # --- 1. Exact hash lookup ---
        entry = self._store.get(key)
        if entry is not None:
            elapsed = time.monotonic() - entry.created_at
            if elapsed > self._ttl_seconds:
                del self._store[key]
            else:
                entry.hit_count += 1
                self._hits += 1
                logger.info("Cache HIT (exact) for query key={key}", key=key)
                return entry.response

        if not self._semantic_enabled:
            self._misses += 1
            logger.debug("Cache MISS for query key={key}", key=key)
            return None

        # --- 2. Sentence-embedding similarity (primary) ---
        query_vec = self._embed_query(normalized)
        if query_vec is not None:
            sem_key = self._find_embedding_match(query_vec)
            if sem_key is not None:
                sem_entry = self._store.get(sem_key)
                if sem_entry is not None and (time.monotonic() - sem_entry.created_at) <= self._ttl_seconds:
                    sem_entry.hit_count += 1
                    self._hits += 1
                    self._semantic_hits += 1
                    self._embed_hits += 1
                    logger.info("Cache HIT (embedding) for query key={key}", key=sem_key)
                    return sem_entry.response
        else:
            # --- 3. TF-IDF similarity fallback ---
            tfidf_key = self._find_tfidf_match(normalized)
            if tfidf_key is not None:
                tfidf_entry = self._store.get(tfidf_key)
                if tfidf_entry is not None and (time.monotonic() - tfidf_entry.created_at) <= self._ttl_seconds:
                    tfidf_entry.hit_count += 1
                    self._hits += 1
                    self._semantic_hits += 1
                    self._tfidf_hits += 1
                    logger.info("Cache HIT (TF-IDF fallback) for query key={key}", key=tfidf_key)
                    return tfidf_entry.response

        self._misses += 1
        logger.debug("Cache MISS for query key={key}", key=key)
        return None

    def put(self, query: str, response: AgentResponse) -> None:
        """Store a response in the cache, pre-computing the embedding vector for future lookups."""
        if not self._enabled:
            return

        normalized = self._normalize_query(query)
        key = self._hash_key(normalized)

        # Pre-compute embedding so it's ready for future similarity comparisons
        embedding = self._embed_query(normalized) if self._semantic_enabled else None
        self._store[key] = CacheEntry(response, normalized_query=normalized, embedding=embedding)
        embed_status = "computed" if embedding is not None else "skipped"
        logger.debug("Cache PUT for query key={key} (embedding={status})", key=key, status=embed_status)

    def invalidate(self, query: str) -> bool:
        """Remove a specific query from the cache. Returns True if entry existed."""
        normalized = self._normalize_query(query)
        key = self._hash_key(normalized)
        removed = self._store.pop(key, None) is not None
        if removed:
            logger.debug("Cache INVALIDATED for query key={key}", key=key)
        return removed

    def clear(self) -> None:
        """Clear all cached entries."""
        count = len(self._store)
        self._store.clear()
        logger.info("Cache CLEARED ({count} entries removed)", count=count)

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics with semantic backend breakdown."""
        total = self._hits + self._misses
        return {
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "semantic_hits": self._semantic_hits,
            "embedding_hits": self._embed_hits,
            "tfidf_hits": self._tfidf_hits,
            "hit_rate": round(self._hits / max(1, total), 4),
            "semantic_enabled": self._semantic_enabled,
            "semantic_threshold": self._semantic_threshold,
            "embedding_backend": "fastembed" if _FASTEMBED_AVAILABLE else ("tfidf" if _SKLEARN_AVAILABLE else "none"),
        }
