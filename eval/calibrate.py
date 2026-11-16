"""Choisir le seuil « hors sujet » (MIN_RERANK_SCORE) à partir des données, pas au hasard.

On compare la meilleure note du reranker pour :
- les 52 questions du jeu de test (dans le sujet : l'assistant DOIT répondre) ;
- des questions hors sujet (l'assistant doit refuser).
Puis on affiche, pour plusieurs seuils, combien de questions de chaque groupe seraient bloquées.

Usage (dans le conteneur) :
    python -m eval.calibrate
"""
import json

from eval.evaluate import QUESTIONS_FILE
from src.rag import retrieve

OUT_OF_SCOPE = [
    "Quel est le prix d'un ticket de métro à Paris ?",
    "Quel temps fera-t-il demain à Orléans ?",
    "Donne-moi une recette de crêpes",
    "Comment renouveler mon passeport ?",
    "Quelle est la capitale de l'Australie ?",
    "Comment déclarer mes impôts quand je suis retraité ?",
    "Comment créer une SARL ?",
    "Qui a gagné la Coupe du monde 2018 ?",
    "Comment adopter un chat dans un refuge ?",
    "Quels vaccins faut-il pour partir en Thaïlande ?",
    "Comment contester une amende de stationnement ?",
    "Combien coûte une carte grise ?",
]


def best_score(question: str) -> float:
    passages = retrieve(question, k=1, use_reranker=True)
    return passages[0]["score"] if passages else float("-inf")


def main() -> None:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        in_scope = [json.loads(line)["question"] for line in f]

    s_in = sorted(((best_score(q), q) for q in in_scope))
    s_out = sorted(((best_score(q), q) for q in OUT_OF_SCOPE), reverse=True)

    print("\nQuestions DANS le sujet avec les notes les plus basses (risque d'être bloquées) :")
    for s, q in s_in[:5]:
        print(f"  {s:6.2f}  {q}")
    print("\nQuestions HORS sujet avec les notes les plus hautes (risque de passer) :")
    for s, q in s_out[:5]:
        print(f"  {s:6.2f}  {q}")

    print(f"\n{'Seuil':>7}  {'dans le sujet bloquées':>24}  {'hors sujet bloquées':>21}")
    for t in [x / 2 for x in range(-12, 5)]:  # de -6 à +2 par pas de 0,5
        blocked_in = sum(s < t for s, _ in s_in)
        blocked_out = sum(s < t for s, _ in s_out)
        print(f"{t:>7.1f}  {blocked_in:>15}/{len(s_in):<8}  {blocked_out:>13}/{len(s_out)}")

    print("\nChoisis le seuil le plus haut qui ne bloque AUCUNE question dans le sujet "
          "(avec un peu de marge), puis mets-le dans MIN_RERANK_SCORE (src/config.py).")


if __name__ == "__main__":
    main()
