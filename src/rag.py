"""Le cœur du RAG : retrouver les bons passages, puis faire rédiger la réponse au LLM."""
from src import bm25, config
from src.index import embed_query, get_collection
from src.llm import chat

SYSTEM_PROMPT = """Tu es ÉtuDroits, un assistant qui aide les étudiants et les stagiaires \
en France à comprendre leurs droits et leurs démarches.

Règles :
- Réponds UNIQUEMENT à partir des extraits fournis, jamais de tes connaissances.
- Commence par une réponse directe en une phrase (oui / non / le montant / la démarche), \
puis donne les détails utiles.
- Les extraits peuvent décrire plusieurs situations (étudiant, élève, apprenti, \
fonction publique…). Réponds pour la situation de la personne. Si elle ne la précise pas, \
considère qu'il s'agit d'un étudiant de l'enseignement supérieur, et ne mentionne les \
autres cas que s'ils changent la réponse, en une ligne.
- Ignore les extraits qui ne concernent pas la question.
- Cite tes sources avec leur numéro entre crochets, par exemple [1] ou [2][3].
- Donne les montants, durées et conditions exactement comme dans les extraits.
- Si les extraits ne permettent pas de répondre, dis-le clairement et propose \
de consulter service-public.gouv.fr ou un organisme compétent (Crous, CAF, université).
- Réponds en français, de façon claire et concise (listes à puces si utile)."""

OUT_OF_SCOPE_ANSWER = (
    "Je ne trouve pas d'information sur ce sujet dans mes fiches. Je réponds aux questions "
    "sur les droits et démarches des étudiants, stagiaires et apprentis (stages, bourses, "
    "logement, alternance, études, international). Pour le reste, consultez "
    "service-public.gouv.fr."
)

_collection = None


# ------------------------------------------------------------ les deux recherches
def dense_search(question: str, n: int) -> list:
    """Recherche par le sens (embeddings + ChromaDB)."""
    global _collection
    if _collection is None:
        _collection = get_collection()
    res = _collection.query(query_embeddings=[embed_query(question)], n_results=n)
    return [
        {"id": i, "text": doc, "score": 1 - dist, **meta}
        for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                      res["metadatas"][0], res["distances"][0])
    ]


def rrf_fusion(rankings: list, k: int = config.RRF_K) -> list:
    """Reciprocal Rank Fusion : combine plusieurs classements en un seul.

    Chaque passage gagne 1 / (k + rang) dans chaque classement où il apparaît.
    Un passage bien classé par les deux recherches finit donc en tête.
    """
    scores, passages = {}, {}
    for ranking in rankings:
        for rank, p in enumerate(ranking, start=1):
            scores[p["id"]] = scores.get(p["id"], 0) + 1 / (k + rank)
            passages.setdefault(p["id"], p)
    best = sorted(scores, key=scores.get, reverse=True)
    return [{**passages[i], "score": scores[i]} for i in best]


def diversify(passages: list, k: int, max_per_fiche) -> list:
    """Garde les `k` meilleurs passages, avec au plus `max_per_fiche` passages par fiche."""
    if not max_per_fiche:
        return passages[:k]
    kept, per_fiche = [], {}
    for p in passages:
        if per_fiche.get(p["fiche_id"], 0) < max_per_fiche:
            kept.append(p)
            per_fiche[p["fiche_id"]] = per_fiche.get(p["fiche_id"], 0) + 1
        if len(kept) == k:
            break
    return kept


def retrieve(question: str, k: int = config.TOP_K, mode: str = None,
             use_reranker: bool = None) -> list:
    """mode : "dense" (sens), "bm25" (mots-clés) ou "hybrid" (les deux, fusionnés).
    use_reranker : relire les meilleurs candidats avec le reranker (défaut : config)."""
    mode = mode or config.SEARCH_MODE
    if use_reranker is None:
        use_reranker = config.USE_RERANKER
    n = max(k * 4, config.N_CANDIDATES)  # plus de candidats que nécessaire, pour trier ensuite
    if mode == "dense":
        ranked = dense_search(question, n)
    elif mode == "bm25":
        ranked = bm25.search(question, n)
    else:
        ranked = rrf_fusion([dense_search(question, n), bm25.search(question, n)])
    if use_reranker:
        from src.rerank import rerank  # import ici : le modèle n'est chargé que si on s'en sert
        ranked = rerank(question, ranked[:config.RERANK_CANDIDATES])
    return diversify(ranked, k, config.MAX_PER_FICHE)


# ------------------------------------------------------------ la réponse
def build_messages(question: str, passages: list) -> list:
    context = "\n\n".join(
        f"[{n}] (fiche « {p['title']} », mise à jour {p['date']})\n{p['text']}"
        for n, p in enumerate(passages, start=1)
    )
    user = f"Extraits :\n\n{context}\n\nQuestion : {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def is_out_of_scope(passages: list) -> bool:
    """Hors sujet si même le meilleur passage a une note du reranker trop basse."""
    if not config.USE_RERANKER or config.MIN_RERANK_SCORE is None:
        return False
    return not passages or passages[0]["score"] < config.MIN_RERANK_SCORE


def answer(question: str, k: int = config.TOP_K) -> dict:
    passages = retrieve(question, k)
    if is_out_of_scope(passages):
        # Pas d'appel au LLM : on économise le quota et on n'affiche pas de sources hors sujet
        return {"question": question, "answer": OUT_OF_SCOPE_ANSWER, "sources": [],
                "provider": "aucun (hors sujet)"}
    text, provider = chat(build_messages(question, passages))
    return {"question": question, "answer": text, "sources": passages, "provider": provider}
