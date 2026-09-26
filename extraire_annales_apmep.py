#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraire_annales_apmep.py
==========================

Construit un fichier "seed" (même esprit que seed_problems.csv) à partir de
vraies annales du Brevet des collèges publiées par l'APMEP
(https://www.apmep.fr/Annales-du-Brevet-des-colleges), en PDF, sujet et
corrigé étant deux fichiers séparés.

POURQUOI CE SCRIPT EXISTE : l'objectif de Jérôme est un dataset dont les
réponses ne contiennent AUCUNE hallucination. La réponse correcte de chaque
exercice n'est donc pas inventée ni calculée par un modèle : elle est extraite
directement du texte du corrigé officiel. Le corrigé complet est aussi
conservé (colonne "corrige_officiel") pour servir de texte de référence
("grounding") au modèle qui rédigera ensuite le dialogue pédagogique, dans
generate_from_annales.py.

CONTRAINTE RÉSEAU : ce script tourne dans un environnement qui n'a pas accès
à apmep.fr (site bloqué en sortie réseau ici, et son robots.txt renvoyait une
erreur 503 au moment où ce script a été écrit). Il faut donc TÉLÉCHARGER les
PDF toi-même (depuis un navigateur normal, ça fonctionne) et les déposer dans
un dossier, avec une convention de nom simple pour que le script puisse
associer chaque sujet à son corrigé sans ambiguïté :

    <identifiant_session>__sujet.pdf
    <identifiant_session>__corrige.pdf

Exemple :
    annales-source/2024-07-metropole__sujet.pdf
    annales-source/2024-07-metropole__corrige.pdf
    annales-source/2024-09-antilles__sujet.pdf
    annales-source/2024-09-antilles__corrige.pdf

(les noms de fichiers d'origine sur apmep.fr ne suivent pas une convention
assez régulière pour être associés automatiquement de façon fiable — voir le
message envoyé à l'utilisateur pour des liens de départ)

LIMITE IMPORTANTE (contrôle qualité) : pour un exercice à plusieurs
sous-questions, la "réponse correcte" retenue automatiquement est une
heuristique (les 2 dernières phrases du corrigé de l'exercice) : elle peut ne
capturer que la conclusion de la DERNIÈRE sous-question, pas toutes. Ce n'est
pas un problème pour le contrôle qualité automatique de generate_from_annales.py
(qui vérifie que cette conclusion précise apparaît bien dans le dialogue final),
mais une relecture humaine reste recommandée, en particulier sur les exercices
à tiroirs (plusieurs questions indépendantes).

Installation :
    pip install pdfplumber

Utilisation :
    python3 extraire_annales_apmep.py --source annales-source --out annales_apmep.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Extraction du texte des PDF
# ---------------------------------------------------------------------------

def extraire_texte_pdf(chemin) -> str:
    import pdfplumber

    morceaux = []
    with pdfplumber.open(chemin) as pdf:
        for page in pdf.pages:
            morceaux.append(page.extract_text() or "")
    return "\n".join(morceaux)


# ---------------------------------------------------------------------------
# 2. Découpage en exercices
# ---------------------------------------------------------------------------

MOTIF_EXERCICE = re.compile(r"Exercice\s+(\d+)\b", re.IGNORECASE)


def decouper_en_exercices(texte: str) -> dict:
    """Renvoie {numéro_exercice (str) : texte_du_bloc}. Si un même numéro
    apparaît plusieurs fois (rare, mais possible avec du texte mal extrait),
    les occurrences sont concaténées plutôt qu'écrasées."""
    positions = [(m.start(), m.group(1)) for m in MOTIF_EXERCICE.finditer(texte)]
    blocs: dict = {}
    for i, (debut, numero) in enumerate(positions):
        fin = positions[i + 1][0] if i + 1 < len(positions) else len(texte)
        bloc = texte[debut:fin].strip()
        blocs[numero] = (blocs.get(numero, "") + "\n" + bloc).strip() if numero in blocs else bloc
    return blocs


# ---------------------------------------------------------------------------
# 3. Réponse finale (heuristique) et thème (mots-clés)
# ---------------------------------------------------------------------------

MOTIF_NOMBRE = r"[-−]?\d+(?:[.,]\d+)?"


def extraire_reponse_courte(texte_corrige_exercice: str) -> str:
    """Heuristique : cherche le dernier "= <nombre>" du corrigé (une
    conclusion de calcul se termine presque toujours ainsi), et à défaut le
    dernier nombre présent dans le texte. Renvoie UN SEUL nombre plutôt
    qu'une phrase entière : si on gardait toute la phrase de conclusion (avec
    ses calculs intermédiaires), le contrôle qualité de generate_from_annales.py
    exigerait à tort que TOUS ces nombres intermédiaires réapparaissent dans
    la conclusion du dialogue généré, ce qui rejetterait des dialogues
    parfaitement corrects. Sur un exercice à plusieurs sous-questions, ne
    capture que la conclusion de la DERNIÈRE sous-question (voir la remarque
    en tête de fichier)."""
    matches_egal = list(re.finditer(r"=\s*(" + MOTIF_NOMBRE + r")", texte_corrige_exercice))
    if matches_egal:
        return matches_egal[-1].group(1)
    matches_nombre = list(re.finditer(MOTIF_NOMBRE, texte_corrige_exercice))
    if matches_nombre:
        return matches_nombre[-1].group(0)
    phrases = re.split(r"(?<=[.!?])\s+", texte_corrige_exercice.strip())
    phrases = [p.strip() for p in phrases if p.strip()]
    return phrases[-1] if phrases else texte_corrige_exercice.strip()


MOTS_CLES_THEME = [
    (r"pythagore", "pythagore"),
    (r"thal[eè]s", "thales"),
    (r"proportionnalit\w*|proportionnel\w*", "proportionnalite"),
    (r"probabilit\w*", "probabilites"),
    (r"fonctions?\b", "fonctions"),
    (r"p[ée]rim[eè]tres?\b|\baires?\b|volumes?\b", "aires_perimetres_volumes"),
    (r"[ée]quations?\b|in[ée]quations?\b", "equations"),
    (r"pourcentages?\b", "pourcentages"),
    (r"statistiques?\b|moyennes?\b|m[ée]dianes?\b", "statistiques"),
    (r"vecteurs?\b", "vecteurs"),
    (r"puissances?\b", "puissances"),
    (r"nombres?\s+relatifs?\b", "nombres_relatifs"),
    (r"racines?\s+carr[ée]es?\b", "racines_carrees"),
    (r"trigonom[ée]trie\b|cosinus\b|sinus\b", "trigonometrie"),
]
MOTS_CLES_THEME = [(re.compile(motif, re.IGNORECASE), theme) for motif, theme in MOTS_CLES_THEME]


def deviner_theme(enonce: str) -> str:
    """Classification par mots-clés (avec limites de mot, pour éviter les faux
    positifs du type "aire" détecté dans "supplémentaires"), volontairement
    simple et déterministe (pas d'appel à un modèle : le thème n'est qu'une
    étiquette de tri, mais autant éviter tout risque inutile). Renvoie
    "a_classifier" si rien ne correspond — à corriger à la main si besoin."""
    for motif, theme in MOTS_CLES_THEME:
        if motif.search(enonce):
            return theme
    return "a_classifier"


# ---------------------------------------------------------------------------
# 4. Association sujet / corrigé et construction du CSV
# ---------------------------------------------------------------------------

def trouver_paires(dossier: Path) -> dict:
    """Associe chaque <id>__sujet.pdf à son <id>__corrige.pdf (même dossier,
    convention de nom imposée — voir l'en-tête de ce fichier)."""
    sujets = {p.stem[:-len("__sujet")]: p for p in dossier.glob("*__sujet.pdf")}
    corriges = {p.stem[:-len("__corrige")]: p for p in dossier.glob("*__corrige.pdf")}

    paires = {}
    manquants = []
    for session_id, chemin_sujet in sujets.items():
        if session_id in corriges:
            paires[session_id] = (chemin_sujet, corriges[session_id])
        else:
            manquants.append(f"{session_id} : sujet présent mais corrigé absent")
    for session_id in corriges:
        if session_id not in sujets:
            manquants.append(f"{session_id} : corrigé présent mais sujet absent")

    if manquants:
        print("Avertissement, sessions incomplètes ignorées :")
        for m in manquants:
            print(f"  - {m}")

    return paires


def construire_lignes(session_id: str, chemin_sujet: Path, chemin_corrige: Path) -> list:
    texte_sujet = extraire_texte_pdf(chemin_sujet)
    texte_corrige = extraire_texte_pdf(chemin_corrige)

    exercices_sujet = decouper_en_exercices(texte_sujet)
    exercices_corrige = decouper_en_exercices(texte_corrige)

    lignes = []
    for numero, enonce in exercices_sujet.items():
        corrige = exercices_corrige.get(numero)
        if corrige is None:
            print(f"  [ignoré] {session_id} exercice {numero} : pas de corrigé correspondant trouvé")
            continue
        lignes.append({
            "id": f"apmep-{session_id}-ex{numero}",
            "niveau": "college",
            "theme": deviner_theme(enonce),
            "enonce": enonce,
            "reponse_correcte": extraire_reponse_courte(corrige),
            "corrige_officiel": corrige,
            "erreur_typique": "",  # laissé vide : proposé par le modèle au moment de la génération
            "source": "apmep",
            "source_id": f"{session_id}-ex{numero}",
            "licence_source": "APMEP (usage pédagogique — voir apmep.fr pour les conditions exactes)",
        })
    return lignes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="annales-source", help="Dossier contenant les PDF <id>__sujet.pdf / <id>__corrige.pdf")
    parser.add_argument("--out", default="annales_apmep.csv", help="Fichier CSV de sortie")
    args = parser.parse_args()

    dossier = Path(args.source)
    if not dossier.is_dir():
        sys.exit(f"Dossier introuvable : {dossier}")

    paires = trouver_paires(dossier)
    if not paires:
        sys.exit(
            f"Aucune paire <id>__sujet.pdf / <id>__corrige.pdf trouvée dans {dossier}. "
            "Vérifie la convention de nommage (voir l'en-tête de ce script)."
        )

    toutes_les_lignes = []
    for session_id, (chemin_sujet, chemin_corrige) in sorted(paires.items()):
        print(f"[traitement] session {session_id}")
        lignes = construire_lignes(session_id, chemin_sujet, chemin_corrige)
        toutes_les_lignes.extend(lignes)
        print(f"  -> {len(lignes)} exercice(s) extrait(s) avec réponse")

    if not toutes_les_lignes:
        sys.exit("Aucun exercice extrait (voir les avertissements ci-dessus).")

    colonnes = ["id", "niveau", "theme", "enonce", "reponse_correcte", "corrige_officiel",
                "erreur_typique", "source", "source_id", "licence_source"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colonnes)
        writer.writeheader()
        writer.writerows(toutes_les_lignes)

    print(f"\nTerminé : {len(toutes_les_lignes)} exercices écrits dans {args.out}")
    print("Relis un échantillon de 'reponse_correcte' à la main avant de lancer generate_from_annales.py :")
    print("l'extraction est fiable sur des exercices simples, moins sur les exercices à tiroirs.")


if __name__ == "__main__":
    main()
