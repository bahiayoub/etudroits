"""Découpage des fiches en passages (chunks) pour la recherche.

Chaque section de fiche (ex. « Gratification > Stage dans une entreprise privée »)
devient un ou plusieurs passages. Le titre de la fiche et de la section est répété
en tête de chaque passage pour qu'il reste compréhensible tout seul.
"""


def split_words(text: str, size: int, overlap: int) -> list:
    """Coupe un texte en morceaux d'au plus `size` mots, avec `overlap` mots en commun."""
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_fiche(fiche: dict, size: int, overlap: int) -> list:
    chunks = []
    for s_idx, section in enumerate(fiche["sections"]):
        header = fiche["title"] + (f" > {section['heading']}" if section["heading"] else "")
        for c_idx, piece in enumerate(split_words(section["text"], size, overlap)):
            chunks.append({
                "id": f"{fiche['id']}-{s_idx}-{c_idx}",
                "text": f"{header}\n{piece}",
                "fiche_id": fiche["id"],
                "title": fiche["title"],
                "section": section["heading"],
                "url": fiche["url"],
                "date": fiche["date"],
            })
    return chunks
