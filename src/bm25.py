"""Recherche par mots-clés (BM25), le complément de la recherche par le sens.

BM25 est l'algorithme classique des moteurs de recherche : il donne un bon score à un
passage qui contient les mots de la question, surtout si ces mots sont rares dans
l'ensemble des passages (« colocation » compte plus que « étudiant »).

Les mots sont normalisés avant comparaison : minuscules, sans accents, sans mots vides
(« le », « de »…) et réduits à leur racine (« alternant » et « alternance » → « altern »).
"""
import json
import re
import unicodedata

import snowballstemmer
from rank_bm25 import BM25Okapi

from src import config

STOPWORDS = set("""
a ai au aux avec ce ces cet cette d dans de des du elle en est et etre il ils je j l la le les
leur lui m ma mais me mes mon ne nous on ou par pas pour qu que qui s sa se ses si son sur t ta
te tes ton tu un une vos votre vous y peut peux puis faut quoi comment est-ce
""".split())

_stemmer = snowballstemmer.stemmer("french")
_index = None  # (bm25, chunks)


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list:
    words = re.findall(r"\w+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return [strip_accents(w) for w in _stemmer.stemWords(words)]


def _load():
    global _index
    if _index is None:
        with open(config.CHUNKS_FILE, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        _index = (BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks)
    return _index


def search(question: str, n: int) -> list:
    """Renvoie les `n` passages qui partagent le plus de mots-clés avec la question."""
    bm25, chunks = _load()
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:n]
    return [{**chunks[i], "score": float(scores[i])} for i in best if scores[i] > 0]
