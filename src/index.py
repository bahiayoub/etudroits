"""Étape 3 : découper les fiches, calculer les embeddings et les ranger dans ChromaDB.

Usage :
    python -m src.index
"""
import json

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src import config
from src.chunking import chunk_fiche

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_passages(texts: list) -> list:
    # Les modèles E5 attendent le préfixe "passage: " pour les documents
    return get_model().encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                              batch_size=32, show_progress_bar=True).tolist()


def embed_query(text: str) -> list:
    # ... et "query: " pour les questions
    return get_model().encode(f"query: {text}", normalize_embeddings=True).tolist()


def get_collection(create: bool = False):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if create:
        try:
            client.delete_collection(config.COLLECTION)
        except Exception:
            pass
        return client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    return client.get_collection(config.COLLECTION)


def main() -> None:
    with open(config.FICHES_FILE, encoding="utf-8") as f:
        fiches = [json.loads(line) for line in f]

    chunks = [c for fiche in fiches
              for c in chunk_fiche(fiche, config.CHUNK_SIZE, config.CHUNK_OVERLAP)]
    print(f"{len(fiches)} fiches → {len(chunks)} passages")

    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    embeddings = embed_passages([c["text"] for c in chunks])

    collection = get_collection(create=True)
    batch = 1000
    for i in tqdm(range(0, len(chunks), batch), desc="Indexation"):
        part = chunks[i:i + batch]
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=embeddings[i:i + batch],
            metadatas=[{k: c[k] for k in ("fiche_id", "title", "section", "url", "date")}
                       for c in part],
        )
    print(f"Index prêt : collection '{config.COLLECTION}' ({collection.count()} passages)")


if __name__ == "__main__":
    main()
