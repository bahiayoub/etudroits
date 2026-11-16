"""Configuration centrale du projet ÉtuDroits."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "vosdroits-latest.zip"
FICHES_FILE = DATA_DIR / "fiches.jsonl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"

# --- Source : fiches "Vos droits et démarches" de Service-Public (DILA, Licence Ouverte) ---
VOSDROITS_URL = "https://lecomarquage.service-public.fr/vdd/3.4/part/zip/vosdroits-latest.zip"

# Rubriques du site gardées entièrement (identifiants du fil d'Ariane)
CATEGORIES = {
    "N20398": "Bourses et aides pour étudiant",
    "N58": "Inscription dans l'enseignement supérieur",
    "N13511": "Étudier à l'étranger",
    "N31850": "Stage en entreprise",
    "N11240": "Alternance",
    "N31346": "Étudiant étranger en France",
    "N20360": "Aides personnelles au logement",
}

# Fiches gardées aussi si leur titre contient un de ces mots
TITLE_KEYWORDS = re.compile(
    r"\b(stages?|stagiaires?|étudiant(e|s|es)?|bourses?|crous|apprenti(s|ssage)?|"
    r"alternance|erasmus|parcoursup|césure|service civique)\b",
    re.IGNORECASE,
)

# Fiches attrapées par un mot-clé mais hors sujet (retirées après relecture manuelle)
EXCLUDED_IDS = {
    # fonctionnaire stagiaire (pas un stage étudiant)
    "F18933", "F13825", "F1895", "F31264", "F1564",
    # faux amis : stage de sécurité routière, apprentissage de la conduite
    "F14208", "F2826",
    # bourses d'école, de collège et de lycée
    "F984", "F616", "F1883", "F32928",
    # autres publics : moins de 16 ans, au pair, salarié détaché ICT, reconversion, organismes
    "F1882", "F13348", "F35178", "F13516", "F15854",
}

# --- Découpage (chunking) ---
CHUNK_SIZE = 250     # en mots
CHUNK_OVERLAP = 40   # en mots

# --- Embeddings (multilingue, tourne sur CPU) ---
EMBED_MODEL = "intfloat/multilingual-e5-small"
COLLECTION = f"fiches_c{CHUNK_SIZE}_o{CHUNK_OVERLAP}"

# --- Recherche ---
TOP_K = 5
MAX_PER_FICHE = 2    # au plus 2 passages d'une même fiche dans les résultats (None = pas de limite)
SEARCH_MODE = "hybrid"  # "dense" (sens), "bm25" (mots-clés) ou "hybrid" (les deux)
N_CANDIDATES = 40    # nombre de candidats demandés à chaque recherche avant fusion et tri
RRF_K = 60           # constante standard de la fusion RRF

# --- Reranker (cross-encoder multilingue, entraîné sur des questions/réponses dont du français) ---
USE_RERANKER = True   # mesuré : Faits@5 79 % → 90 %
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_CANDIDATES = 30  # nombre de candidats relus par le reranker
# Seuil hors sujet : si la meilleure note du reranker est en dessous, on ne répond pas.
# None = désactivé. Valeur à choisir avec : python -m eval.calibrate
MIN_RERANK_SCORE = -3.5  # calibré : 0/52 vraies questions bloquées, 8/12 hors sujet bloquées

# --- LLM : fournisseurs gratuits, essayés dans l'ordre (API au format OpenAI) ---
PROVIDERS = [
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    {
        "name": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
]
