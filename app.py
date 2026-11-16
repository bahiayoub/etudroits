"""Interface web d'ÉtuDroits (Gradio).

En local (dans Docker) :
    docker compose run --rm --service-ports app python app.py
puis ouvrir http://localhost:7860
"""
import gradio as gr

from src.rag import answer

EXAMPLES = [
    "Mon stage dure 3 mois, est-ce qu'on doit me payer ?",
    "Je suis en colocation, est-ce que je peux toucher l'APL ?",
    "Combien est payé un apprenti ?",
    "Comment obtenir une chambre en résidence universitaire Crous ?",
    "Est-ce que je peux faire une année de césure pendant mon master ?",
    "Je suis étudiant étranger hors UE, combien d'heures par an je peux travailler ?",
]

INTRO = """
# 🎓 ÉtuDroits
**L'assistant qui répond à tes questions sur les droits des étudiants, stagiaires et apprentis**,
à partir des fiches officielles de [Service-Public.gouv.fr](https://www.service-public.gouv.fr).
Chaque réponse cite ses sources.
"""

FOOTER = """
---
<small>⚠️ ÉtuDroits ne remplace pas un conseil juridique : vérifie toujours la fiche officielle citée.
Données : fiches « Vos droits et démarches » de Service-Public.gouv.fr (DILA, Licence Ouverte).
Recherche hybride (BM25 + embeddings) avec reranker, réponse rédigée par un LLM open source via Groq.</small>
"""


def format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["### 📚 Sources"]
    for n, p in enumerate(sources, start=1):
        section = f" › {p['section']}" if p["section"] else ""
        lines.append(f"**[{n}]** [{p['title']}{section}]({p['url']})  \n"
                     f"<small>Fiche mise à jour le {p['date']}</small>")
    return "\n\n".join(lines)


def ask(question: str):
    question = (question or "").strip()
    if not question:
        return "Pose une question 🙂", ""
    try:
        res = answer(question)
    except Exception as e:  # ex. clé API absente ou quota gratuit épuisé
        return f"❌ Une erreur est survenue : {e}", ""
    return res["answer"], format_sources(res["sources"])


with gr.Blocks(title="ÉtuDroits") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        question = gr.Textbox(label="Ta question", scale=5, lines=2,
                              placeholder="Ex. : Mon stage dure 3 mois, est-ce qu'on doit me payer ?")
        submit = gr.Button("Demander", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=question, label="Exemples de questions")
    response = gr.Markdown(label="Réponse")
    sources = gr.Markdown()
    gr.Markdown(FOOTER)

    submit.click(ask, inputs=question, outputs=[response, sources])
    question.submit(ask, inputs=question, outputs=[response, sources])


if __name__ == "__main__":
    # 0.0.0.0 : accessible depuis l'extérieur du conteneur Docker
    demo.launch(server_name="0.0.0.0", server_port=7860)
