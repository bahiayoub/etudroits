"""Poser une question à ÉtuDroits depuis le terminal.

    python ask.py "Combien doit être payé un stage de 3 mois ?"
    python ask.py                       # mode conversation
    python ask.py --sources-only "..."  # voir seulement les passages retrouvés (sans LLM)
"""
import argparse

from src.rag import answer, retrieve


def show_sources(passages: list) -> None:
    for n, p in enumerate(passages, start=1):
        section = f" > {p['section']}" if p["section"] else ""
        print(f"  [{n}] {p['title']}{section}  (score {p['score']:.2f})\n      {p['url']}")


def run(question: str, sources_only: bool) -> None:
    if sources_only:
        show_sources(retrieve(question))
        return
    res = answer(question)
    print(f"\n{res['answer']}\n")
    if res["sources"]:
        print(f"Sources ({res['provider']}) :")
        show_sources(res["sources"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--sources-only", action="store_true")
    args = parser.parse_args()

    if args.question:
        run(args.question, args.sources_only)
        return
    print("ÉtuDroits : pose ta question (ligne vide pour quitter)")
    while question := input("\n> ").strip():
        run(question, args.sources_only)


if __name__ == "__main__":
    main()
