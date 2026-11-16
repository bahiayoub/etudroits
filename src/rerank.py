"""Reranker : relit chaque passage AVEC la question et les reclasse plus finement.

Différence avec le modèle d'embedding :
- l'embedding résume la question et le passage SÉPARÉMENT en vecteurs, puis on les compare.
  C'est rapide (les passages sont calculés à l'avance) mais approximatif ;
- le reranker (un « cross-encoder ») lit la question et le passage ENSEMBLE et donne un
  score de pertinence. C'est bien plus précis, mais trop lent pour 1 270 passages :
  on ne l'applique qu'aux ~30 meilleurs candidats.
"""
from sentence_transformers import CrossEncoder

from src import config

_model = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANKER_MODEL, max_length=512)
    return _model


def rerank(question: str, passages: list) -> list:
    if not passages:
        return passages
    scores = get_model().predict([(question, p["text"]) for p in passages])
    reranked = [{**p, "score": float(s)} for p, s in zip(passages, scores)]
    return sorted(reranked, key=lambda p: p["score"], reverse=True)
