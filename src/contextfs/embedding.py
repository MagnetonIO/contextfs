"""
Embedding backend for ContextFS.

Supports multiple embedding backends:
- FastEmbed (ONNX-based, faster, recommended)
- SentenceTransformers (PyTorch-based, fallback)

With optional GPU acceleration on Mac (MPS), Linux (CUDA), and Windows (CUDA).
"""

import logging
import os
from abc import ABC, abstractmethod
from enum import Enum

# Disable tokenizers parallelism to avoid deadlocks in multi-threaded contexts
# Must be set before any tokenizers/transformers are imported
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

# Global cache for embedding models (expensive to load)
_embedder_cache: dict[str, "BaseEmbedder"] = {}


class EmbeddingBackend(Enum):
    """Available embedding backends."""

    FASTEMBED = "fastembed"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    AUTO = "auto"  # Auto-detect best available


class DeviceType(Enum):
    """Device types for computation."""

    CPU = "cpu"
    CUDA = "cuda"  # NVIDIA GPU
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"  # Auto-detect best available


class BaseEmbedder(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        """Encode texts to embeddings."""
        pass

    @abstractmethod
    def encode_single(self, text: str) -> list[float]:
        """Encode a single text to embedding."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class FastEmbedder(BaseEmbedder):
    """FastEmbed backend using ONNX runtime."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
        parallel: int | None = None,
    ):
        """
        Initialize FastEmbed backend.

        Args:
            model_name: Model name (will be mapped to FastEmbed model)
            use_gpu: Enable GPU acceleration
            parallel: Number of parallel workers (None = auto, 0 = all cores)
        """
        self._model_name = model_name
        self._use_gpu = use_gpu
        self._parallel = parallel
        self._model = None
        self._dimension = None

    def _ensure_initialized(self) -> None:
        """Lazy initialize the model."""
        if self._model is not None:
            return

        try:
            from fastembed import TextEmbedding

            # Map common model names to FastEmbed models
            model_map = {
                "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
                "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
                "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
            }
            fastembed_model = model_map.get(self._model_name, self._model_name)

            # Configure providers for GPU (with fallback to CPU if GPU fails)
            providers = None
            if self._use_gpu:
                providers = self._get_gpu_providers()
                if providers:
                    logger.info(f"FastEmbed attempting GPU providers: {providers}")

            # Determine parallel workers
            parallel = self._parallel
            if parallel is None:
                # Check if tokenizers parallelism is disabled (indicates threaded context)
                # In this case, disable parallel workers to avoid tokio runtime crashes
                if os.environ.get("TOKENIZERS_PARALLELISM", "").lower() == "false":
                    # Single-threaded to avoid tokio runtime conflicts
                    parallel = 1
                    logger.debug("FastEmbed parallel disabled (TOKENIZERS_PARALLELISM=false)")
                else:
                    # Auto: use half of CPU cores
                    parallel = max(1, (os.cpu_count() or 4) // 2)

            # Try GPU providers first, fall back to CPU if they fail
            try:
                self._model = TextEmbedding(
                    model_name=fastembed_model,
                    providers=providers,
                )
            except Exception as e:
                if providers:
                    logger.warning(f"GPU providers failed ({e}), falling back to CPU")
                    self._model = TextEmbedding(
                        model_name=fastembed_model,
                        providers=None,  # CPU only
                    )
                else:
                    raise
            self._parallel = parallel

            # Get dimension from a test embedding
            test_emb = list(self._model.embed(["test"]))[0]
            self._dimension = len(test_emb)

            provider_info = "GPU" if providers and self._use_gpu else "CPU"
            logger.info(
                f"FastEmbed initialized: {fastembed_model} (dim={self._dimension}, parallel={parallel}, {provider_info})"
            )

        except ImportError:
            raise ImportError(
                "FastEmbed not installed. Install with: pip install fastembed"
                "\nFor GPU support: pip install fastembed-gpu"
            )

    def _get_gpu_providers(self) -> list[str] | None:
        """Get ONNX execution providers for GPU acceleration."""
        import platform

        system = platform.system()

        if system == "Darwin":
            # macOS - try CoreML first, then CPU
            # Note: MPS is not directly supported by ONNX Runtime
            # CoreMLExecutionProvider works on Apple Silicon
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif system == "Linux" or system == "Windows":
            # Try CUDA first, then CPU
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            return None

    def encode(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        """Encode texts to embeddings in batch."""
        self._ensure_initialized()

        if not texts:
            return []

        # Process in configurable batches for efficiency
        # Larger batches = fewer model calls but more memory
        from contextfs.config import get_config

        batch_size = get_config().embedding_batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = list(self._model.embed(batch, parallel=self._parallel))
            all_embeddings.extend([emb.tolist() for emb in embeddings])

        return all_embeddings

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text."""
        result = self.encode([text])
        return result[0] if result else []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._ensure_initialized()
        return self._dimension or 384


class SentenceTransformersEmbedder(BaseEmbedder):
    """SentenceTransformers backend using PyTorch."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
    ):
        """
        Initialize SentenceTransformers backend.

        Args:
            model_name: Model name
            use_gpu: Enable GPU acceleration
        """
        self._model_name = model_name
        self._use_gpu = use_gpu
        self._model = None
        self._dimension = None

    def _ensure_initialized(self) -> None:
        """Lazy initialize the model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            device = self._get_device() if self._use_gpu else "cpu"

            self._model = SentenceTransformer(self._model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()

            logger.info(
                f"SentenceTransformers initialized: {self._model_name} (dim={self._dimension}, device={device})"
            )

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

    def _get_device(self) -> str:
        """Get best available device for PyTorch."""
        import platform

        try:
            import torch

            system = platform.system()

            if system == "Darwin" and torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def encode(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        """Encode texts to embeddings in batch."""
        self._ensure_initialized()

        if not texts:
            return []

        # Process in configurable batches for efficiency
        # Larger batches = fewer model calls but more memory
        from contextfs.config import get_config

        batch_size = get_config().embedding_batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text."""
        self._ensure_initialized()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        self._ensure_initialized()
        return self._dimension or 384


def create_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    backend: EmbeddingBackend | str = EmbeddingBackend.AUTO,
    use_gpu: bool = False,
    parallel: int | None = None,
    use_cache: bool = True,
) -> BaseEmbedder:
    """
    Create an embedder with the specified backend.

    Args:
        model_name: Embedding model name
        backend: Backend to use (fastembed, sentence_transformers, or auto)
        use_gpu: Enable GPU acceleration
        parallel: Number of parallel workers (FastEmbed only)
        use_cache: Use cached embedder if available (default True)

    Returns:
        Configured embedder instance
    """
    if isinstance(backend, str):
        backend = EmbeddingBackend(backend)

    if backend == EmbeddingBackend.AUTO:
        # Try FastEmbed first (faster), fall back to SentenceTransformers
        try:
            import fastembed  # noqa: F401

            backend = EmbeddingBackend.FASTEMBED
            logger.info("Auto-selected FastEmbed backend")
        except ImportError:
            backend = EmbeddingBackend.SENTENCE_TRANSFORMERS
            logger.info("Auto-selected SentenceTransformers backend (FastEmbed not available)")

    # Create cache key
    cache_key = f"{backend.value}:{model_name}:{use_gpu}"

    # Return cached embedder if available
    if use_cache and cache_key in _embedder_cache:
        logger.debug(f"Using cached embedder: {cache_key}")
        return _embedder_cache[cache_key]

    # Create new embedder
    if backend == EmbeddingBackend.FASTEMBED:
        embedder = FastEmbedder(
            model_name=model_name,
            use_gpu=use_gpu,
            parallel=parallel,
        )
    else:
        embedder = SentenceTransformersEmbedder(
            model_name=model_name,
            use_gpu=use_gpu,
        )

    # Cache it
    if use_cache:
        _embedder_cache[cache_key] = embedder
        logger.debug(f"Cached new embedder: {cache_key}")

    return embedder


def check_gpu_available() -> dict:
    """
    Check GPU availability for embedding acceleration.

    Returns:
        Dict with GPU status information
    """
    import platform

    result = {
        "system": platform.system(),
        "gpu_available": False,
        "gpu_type": None,
        "recommended_backend": "fastembed",
    }

    system = platform.system()

    # Check CUDA
    try:
        import torch

        if torch.cuda.is_available():
            result["gpu_available"] = True
            result["gpu_type"] = "cuda"
            result["cuda_device"] = torch.cuda.get_device_name(0)
            return result
    except ImportError:
        pass

    # Check MPS (Apple Silicon)
    if system == "Darwin":
        try:
            import torch

            if torch.backends.mps.is_available():
                result["gpu_available"] = True
                result["gpu_type"] = "mps"
                return result
        except (ImportError, AttributeError):
            pass

        # Check for CoreML (works with FastEmbed on Apple Silicon)
        try:
            import coremltools  # noqa: F401

            result["gpu_available"] = True
            result["gpu_type"] = "coreml"
            return result
        except ImportError:
            pass

    return result
