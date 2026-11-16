"""Étape 2 : télécharger les fiches Service-Public et garder celles qui concernent
les étudiants et les stagiaires.

Usage :
    python -m src.ingest                 # télécharge le zip officiel puis l'analyse
    python -m src.ingest --zip chemin.zip  # utilise un zip déjà téléchargé
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src import config

DC = "{http://purl.org/dc/elements/1.1/}"
NOTE_TAGS = {"ANoter": "À noter", "ASavoir": "À savoir", "Attention": "Attention",
             "Exemple": "Exemple", "Rappel": "Rappel"}


# ---------------------------------------------------------------- téléchargement
def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60,
                      headers={"User-Agent": "etudroits-student-project"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc="Téléchargement") as bar:
            for block in r.iter_content(chunk_size=1 << 16):
                f.write(block)
                bar.update(len(block))
    return dest


# ---------------------------------------------------------------- lecture du XML
def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


BLOCK_TAGS = {"Paragraphe", "Titre", "Item", "Liste", "Cellule", "Rangée"}
SKIP_TAGS = {"Condition", "ServiceEnLigne", "OuSAdresser", "Source"}


def _raw_text(elem) -> str:
    parts = [elem.text or ""]
    for child in elem:
        sep = " " if child.tag in BLOCK_TAGS else ""  # espace entre blocs, pas dans "309<Exposant>e</Exposant>"
        parts += [sep, _raw_text(child), sep, child.tail or ""]
    return "".join(parts)


def text_of(elem) -> str:
    return clean(_raw_text(elem)) if elem is not None else ""


def render(elem, path: list, sections: dict) -> None:
    """Parcourt le contenu d'une fiche et range chaque ligne sous son titre de section."""

    def add(line: str) -> None:
        if line:
            sections.setdefault(" > ".join(path), []).append(line)

    for child in elem:
        tag = child.tag
        if tag in SKIP_TAGS:
            continue
        if tag in ("Chapitre", "SousChapitre", "Cas", "Situation"):
            title = text_of(child.find("Titre"))
            render(child, path + [title] if title else path, sections)
        elif tag == "Titre":
            continue  # déjà utilisé comme titre de section
        elif tag == "Paragraphe":
            add(text_of(child))
        elif tag == "Liste":
            for item in child.findall("Item"):
                add("- " + text_of(item))
        elif tag == "Tableau":
            add(text_of(child.find("Titre")))
            for row in child.iter("Rangée"):
                add(" | ".join(text_of(c) for c in row.findall("Cellule")))
        elif tag in NOTE_TAGS:
            label = text_of(child.find("Titre")) or NOTE_TAGS[tag]
            body = " ".join(text_of(c) for c in child if c.tag != "Titre")
            add(f"{label} : {body}")
        else:
            render(child, path, sections)


def parse_fiche(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    sections: dict = {}
    # ListeSituations = onglets du site (ex. « Étudiant en France » / « à l'étranger »)
    for part in ("Introduction", "Texte", "ListeSituations"):
        node = root.find(part)
        if node is not None:
            render(node, [], sections)
    return {
        "id": root.get("ID"),
        "title": text_of(root.find(f"{DC}title")),
        "description": text_of(root.find(f"{DC}description")),
        "date": text_of(root.find(f"{DC}date")).replace("modified ", ""),
        "url": root.get("spUrl") or f"https://www.service-public.gouv.fr/particuliers/vosdroits/{root.get('ID')}",
        "breadcrumb": [n.get("ID") for n in root.iter("Niveau")],
        "sections": [{"heading": h, "text": "\n".join(lines)} for h, lines in sections.items()],
    }


def is_relevant(fiche: dict) -> bool:
    if fiche["title"].startswith("["):  # fragments techniques réutilisables, pas de vraies fiches
        return False
    if fiche["id"] in config.EXCLUDED_IDS:
        return False
    in_category = any(n in config.CATEGORIES for n in fiche["breadcrumb"])
    return in_category or bool(config.TITLE_KEYWORDS.search(fiche["title"]))


def extract(zip_path: Path) -> list:
    fiches = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if re.fullmatch(r"(.*/)?F\d+\.xml", n)]
        for name in tqdm(names, desc="Analyse des fiches"):
            try:
                fiche = parse_fiche(z.read(name))
            except ET.ParseError:
                continue
            if is_relevant(fiche) and fiche["sections"]:
                fiches.append(fiche)
    return fiches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, help="zip déjà téléchargé")
    args = parser.parse_args()

    zip_path = args.zip or config.ZIP_PATH
    if not zip_path.exists():
        download(config.VOSDROITS_URL, zip_path)

    fiches = extract(zip_path)
    config.FICHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.FICHES_FILE, "w", encoding="utf-8") as f:
        for fiche in fiches:
            f.write(json.dumps(fiche, ensure_ascii=False) + "\n")
    print(f"{len(fiches)} fiches gardées → {config.FICHES_FILE}")


if __name__ == "__main__":
    main()