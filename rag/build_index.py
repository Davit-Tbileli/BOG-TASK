from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _load_offers(path: Path) -> List[Dict[str, Any]]:

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    load_dotenv()

    repo_root = Path(__file__).resolve().parents[1]
    offers_path = repo_root / "data" / "processed" / "found_offers.json"

    if not offers_path.exists():
        raise FileNotFoundError(f"Offers JSON not found: {offers_path}")

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    config = {
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": os.getenv("COLLECTION_NAME", "bog_offers"),
        "qdrant_distance": os.getenv("QDRANT_DISTANCE", "COSINE"),
        "vector_store": "qdrant",
    }

    offers = _load_offers(offers_path)
    logger.info(f"Loaded {len(offers)} offers from {offers_path}")

    embedder = EmbeddingGenerator(model_name=embedding_model, normalize_embeddings=True)
    store = VectorStore(config=config)

    # Fast fail if Qdrant isn't reachable.
    try:
        store.client.get_collections()
    except Exception as e:
        raise RuntimeError(
            "Qdrant is not reachable. Check QDRANT_URL/QDRANT_API_KEY and that your cluster is running."
        ) from e

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for i, offer in enumerate(offers):
        # Skip malformed stubs if present
        if not isinstance(offer, dict) or not offer.get("details_url"):
            continue

        doc = embedder.preprocess_offer(offer)
        documents.append(doc)

        meta = dict(offer)
        meta["id"] = i
        metadatas.append(meta)

    embeddings = embedder.generate_embeddings(documents)

    # Create collection with correct size (important)
    store.create_collection(vector_size=embedder.embedding_dimension())
    # Ensure payload indexes needed for filtering (category_desc, cities, etc.)
    store.ensure_payload_indexes()
    store.add_documents(documents=documents, embeddings=embeddings, metadata=metadatas)

    logger.info(f"Indexed {len(documents)} offers into Qdrant collection '{store.collection_name}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
