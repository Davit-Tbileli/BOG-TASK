from __future__ import annotations

import os
from dotenv import load_dotenv


def main() -> int:
    """Create Qdrant payload indexes required for filtered search.

    This is fast and does not re-embed or re-upsert documents.
    """

    load_dotenv()

    from rag.vector_store import VectorStore

    config = {
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": os.getenv("COLLECTION_NAME", "bog_offers"),
        "qdrant_distance": os.getenv("QDRANT_DISTANCE", "COSINE"),
        "vector_store": "qdrant",
    }

    store = VectorStore(config=config)
    # This will connect and create indexes on the existing collection.
    store.ensure_payload_indexes()
    print("Payload indexes ensured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
