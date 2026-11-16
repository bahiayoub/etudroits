"""Appel au LLM avec bascule automatique entre fournisseurs gratuits."""
import os

from dotenv import load_dotenv
from openai import OpenAI

from src import config

load_dotenv()


def available_providers() -> list:
    return [p for p in config.PROVIDERS if os.getenv(p["env_key"])]


def chat(messages: list, temperature: float = 0.1) -> tuple:
    """Renvoie (texte de la réponse, nom du fournisseur utilisé)."""
    providers = available_providers()
    if not providers:
        raise RuntimeError("Aucune clé API trouvée : remplis GROQ_API_KEY dans le fichier .env")

    errors = []
    for p in providers:
        client = OpenAI(base_url=p["base_url"], api_key=os.getenv(p["env_key"]),
                        max_retries=1, timeout=60)
        try:
            resp = client.chat.completions.create(model=p["model"], messages=messages,
                                                  temperature=temperature)
            return resp.choices[0].message.content.strip(), p["name"]
        except Exception as e:  # limite atteinte, panne… → fournisseur suivant
            errors.append(f"{p['name']}: {e}")
    raise RuntimeError("Tous les fournisseurs ont échoué :\n" + "\n".join(errors))
