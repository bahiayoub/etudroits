"""Évaluation de la recherche (le « R » du RAG), sans appeler le LLM : gratuit et rapide.

Deux mesures complémentaires :
- Hit@k : une des fiches attendues apparaît-elle parmi les k premiers passages ?
  (la BONNE FICHE est-elle trouvée ?)
- Faits@5 : les 5 passages donnés au LLM contiennent-ils les faits clés de la réponse
  (ex. « 4,50 » pour la gratification) ? (les BONNES SECTIONS sont-elles trouvées ?)

Usage (dans le conteneur) :
    python -m eval.evaluate                          # configuration actuelle
    python -m eval.evaluate --name baseline          # enregistre les résultats sous ce nom
    python -m eval.evaluate --mode bm25              # teste un autre mode de recherche
    python -m eval.evaluate --rerank --name rerank   # active le reranker
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from src import config
from src.rag import retrieve

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
KS = (1, 3, 5)
MAX_K = 10


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).lower()


def first_hit_rank(passages: list, expected: list):
    """Rang (1, 2, 3…) du premier passage qui vient d'une fiche attendue, ou None."""
    for rank, p in enumerate(passages, start=1):
        if p["fiche_id"] in expected:
            return rank
    return None


def facts_found(passages: list, facts: list):
    """True si tous les faits clés sont dans le texte des passages (None si pas de faits)."""
    if not facts:
        return None
    context = normalize(" ".join(p["text"] for p in passages))
    return all(normalize(f) in context for f in facts)


def evaluate(questions: list, mode: str = None, use_reranker: bool = None) -> tuple:
    rows = []
    for q in questions:
        passages = retrieve(q["question"], k=MAX_K, mode=mode, use_reranker=use_reranker)
        rows.append({
            **q,
            "rank": first_hit_rank(passages, q["expected_fiches"]),
            "facts_ok": facts_found(passages[:config.TOP_K], q.get("key_facts", [])),
            "top3": [p["fiche_id"] for p in passages[:3]],
        })

    def summarize(subset: list) -> dict:
        n = len(subset)
        scores = {f"hit@{k}": sum(r["rank"] is not None and r["rank"] <= k for r in subset) / n
                  for k in KS}
        scores["mrr"] = sum(1 / r["rank"] for r in subset if r["rank"]) / n
        with_facts = [r for r in subset if r["facts_ok"] is not None]
        scores["faits@5"] = (sum(r["facts_ok"] for r in with_facts) / len(with_facts)
                             if with_facts else None)
        return scores

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    summary = {"global": summarize(rows), **{c: summarize(rs) for c, rs in sorted(by_cat.items())}}
    return summary, rows


def print_report(summary: dict, rows: list) -> None:
    counts = defaultdict(int)
    for r in rows:
        counts[r["category"]] += 1
    print(f"\n{'':14}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}{'Faits@5':>9}   (n)")
    for name, s in summary.items():
        n = len(rows) if name == "global" else counts[name]
        faits = f"{s['faits@5']:.0%}" if s["faits@5"] is not None else "-"
        print(f"{name:14}{s['hit@1']:>8.0%}{s['hit@3']:>8.0%}{s['hit@5']:>8.0%}"
              f"{s['mrr']:>8.2f}{faits:>9}   ({n})")

    misses = [r for r in rows if r["rank"] is None or r["rank"] > 5]
    print(f"\nBonne fiche absente des 5 premiers passages : {len(misses)}")
    for r in misses:
        rank = r["rank"] or f">{MAX_K}"
        print(f"  [{r['id']}] {r['question']}\n        attendu {r['expected_fiches']}, "
              f"rang {rank}, trouvé en premier : {r['top3']}")

    no_facts = [r for r in rows if r["facts_ok"] is False]
    print(f"\nFaits clés absents des 5 passages donnés au LLM : {len(no_facts)}")
    for r in no_facts:
        print(f"  [{r['id']}] {r['question']}  (attendu : {r['key_facts']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="nom sous lequel enregistrer les résultats")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"],
                        help="mode de recherche (par défaut : celui de config.py)")
    parser.add_argument("--rerank", action="store_true", default=None,
                        help="active le reranker (par défaut : celui de config.py)")
    args = parser.parse_args()

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    summary, rows = evaluate(questions, args.mode, args.rerank)
    print_report(summary, rows)

    if args.name:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"{args.name}.json"
        out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\nRésultats enregistrés → {out}")


if __name__ == "__main__":
    main()
