#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_mathdial.py
======================

Traduit et adapte en français un échantillon du dataset public MathDial
(dialogues professeur/élève en anglais, niveau "7th grade" américain ≈
5e en France), comme matière première pour le fine-tuning.

Ce script fait une TRADUCTION FIDÈLE : il ne change pas le déroulé
pédagogique du dialogue d'origine (les relances du professeur restent
les mêmes, juste traduites). Le renforcement "résistance à la demande
de réponse directe" est ajouté séparément par generate_dataset.py, sur
des dialogues générés à partir de zéro — mélanger les deux dans un seul
script rendrait la traduction moins fidèle.

Licence : MathDial est publié sous licence Creative Commons
Attribution-ShareAlike 4.0 (CC BY-SA 4.0). Toute réutilisation, y
compris ce jeu de données traduit, doit conserver l'attribution et être
partagée sous la même licence. Voir ATTRIBUTION.md (généré à côté du
fichier de sortie) pour la citation exacte à conserver.

Installation :
    pip install openai

Configuration :
    export OPENAI_API_KEY="votre-clé"

Utilisation :
    python3 translate_mathdial.py \
        --source mathdial-source/data/train.jsonl \
        --out mathdial_fr.jsonl \
        --n 50 \
        --model gpt-6-luna
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Réutilise les briques déjà écrites et testées dans generate_dataset.py
# (appel au modèle, extraction du JSON, contrôle qualité automatique).
from generate_dataset import call_llm, extraire_json, valider_dialogue, extraire_nombres


PROMPT_SYSTEME_TRADUCTION = """Tu es un traducteur spécialisé qui adapte un dialogue de tutorat en \
mathématiques, de l'anglais vers le français, pour un usage avec des élèves français.

RÈGLES IMPÉRATIVES :
1. NE MODIFIE PAS le déroulé pédagogique du dialogue d'origine : si le \
professeur donne un indice à un endroit précis, traduis cet indice, ne le \
déplace pas et n'en ajoute pas d'autre. Tu traduis et adaptes, tu n'inventes \
pas de nouveau contenu pédagogique.
2. Remplace le prénom de l'élève par un prénom courant en France (garde-le \
cohérent sur tout le dialogue).
3. Adapte les unités si besoin (dollars -> euros, feet -> mètres) SANS changer \
les valeurs numériques ni la cohérence du calcul — si l'énoncé original utilise \
des pieds ou des dollars, tu peux garder l'unité ("pieds", "dollars") plutôt \
que de recalculer une conversion, l'important est de ne pas fausser les calculs.
4. Le premier message "user" doit combiner : l'énoncé traduit du problème, PUIS \
la solution erronée de l'élève reformulée à la PREMIÈRE PERSONNE, comme si \
l'élève venait de l'écrire lui-même (le champ fourni est à la 3e personne : \
"The student calculated..." -> "J'ai calculé...").
5. Les tours de dialogue suivants (fournis dans "conversation brute") sont à \
traduire dans l'ordre, en gardant qui parle (Teacher -> assistant, Student -> \
user). Les étiquettes entre parenthèses avant le texte du professeur (par \
exemple "(probing)") indiquent le type de relance pédagogique : ne les inclus \
PAS dans le texte traduit, elles servent juste à toi pour bien comprendre \
l'intention de chaque tour et la traduire fidèlement.
6. Le premier message "system" doit présenter un professeur de maths qui ne \
donne jamais la réponse directement (reprends cette idée même si le dialogue \
d'origine ne le formule pas explicitement, c'est le cadre général).

FORMAT DE SORTIE :
Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour :
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
"""

PROMPT_UTILISATEUR_TEMPLATE = """Énoncé du problème (anglais) : {question}

Solution erronée de l'élève, à la 3e personne (anglais) : {solution_erronee}

Profil de l'élève (anglais, pour le contexte seulement, ne pas traduire tel quel \
dans le dialogue) : {profil_eleve}

Réponse numérique correcte attendue : {reponse_correcte}

Conversation brute (anglais, à traduire tour par tour dans l'ordre) :
{conversation_brute}

Traduis et restructure ce dialogue en français au format JSON demandé.
"""


def extraire_reponse_finale(ground_truth: str) -> str:
    """MathDial met la réponse numérique finale seule sur la dernière ligne
    non vide du champ 'ground_truth' (le reste étant le raisonnement)."""
    lignes = [l.strip() for l in ground_truth.strip().splitlines() if l.strip()]
    return lignes[-1] if lignes else ground_truth.strip()


def traduire_entree(entree: dict, model: str, max_essais: int = 2):
    reponse_correcte = extraire_reponse_finale(entree["ground_truth"])
    user_prompt = PROMPT_UTILISATEUR_TEMPLATE.format(
        question=entree["question"],
        solution_erronee=entree["student_incorrect_solution"],
        profil_eleve=entree.get("student_profile", ""),
        reponse_correcte=reponse_correcte,
        conversation_brute=entree["conversation"].replace("|EOM|", "\n"),
    )

    derniere_erreur = ""
    for essai in range(1, max_essais + 1):
        try:
            brut = call_llm(PROMPT_SYSTEME_TRADUCTION, user_prompt, model=model, temperature=0.3)
            dialogue = extraire_json(brut)
            valide, raison = valider_dialogue(dialogue, reponse_correcte, avec_insistance=False)
            if valide:
                dialogue["id"] = f"mathdial-fr-{entree['qid']}"
                dialogue["niveau"] = "college"  # MathDial = "7th grade" US ~ 5e française
                dialogue["theme"] = "probleme_multi_etapes"
                dialogue["source"] = "mathdial"
                dialogue["source_id"] = entree["qid"]
                dialogue["licence_source"] = "CC BY-SA 4.0"
                return dialogue, None
            derniere_erreur = raison
        except Exception as exc:
            derniere_erreur = f"Exception : {exc}"
        time.sleep(1)

    return None, derniere_erreur


ATTRIBUTION_TEXTE = """# Attribution requise (dataset MathDial)

Les dialogues marqués `"source": "mathdial"` dans ce jeu de données sont \
traduits et adaptés à partir de MathDial, publié sous licence \
Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0).
https://creativecommons.org/licenses/by-sa/4.0/

Toute republication de ces dialogues traduits (ou d'un modèle entraîné \
dessus, selon l'interprétation retenue de la clause "ShareAlike") doit :
- citer la source ci-dessous,
- être partagée sous la même licence (CC BY-SA 4.0).

Citation à conserver :
Jakub Macina, Nico Daheim, Sankalan Chowdhury, Tanmay Sinha, Manu Kapur, \
Iryna Gurevych, and Mrinmaya Sachan. 2023. MathDial: A Dialogue Tutoring \
Dataset with Rich Pedagogical Properties Grounded in Math Reasoning \
Problems. In Findings of the Association for Computational Linguistics: \
EMNLP 2023, pages 5602-5621, Singapore. Association for Computational \
Linguistics.

Dépôt d'origine : https://github.com/eth-nlped/mathdial
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="mathdial-source/data/train.jsonl", help="Fichier MathDial d'origine (.jsonl)")
    parser.add_argument("--out", default="mathdial_fr.jsonl", help="Fichier JSONL de sortie (français)")
    parser.add_argument("--n", type=int, default=50, help="Nombre de dialogues à échantillonner et traduire")
    parser.add_argument("--model", default="gpt-6-luna", help="Modèle à utiliser pour la traduction")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour l'échantillonnage (reproductibilité)")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        sys.exit(f"Fichier introuvable : {source_path}. As-tu cloné eth-nlped/mathdial ?")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Variable d'environnement OPENAI_API_KEY absente. "
                  "Fais 'export OPENAI_API_KEY=...' avant de relancer.")

    with open(source_path, encoding="utf-8") as f:
        entrees = [json.loads(l) for l in f]

    random.seed(args.seed)
    echantillon = random.sample(entrees, min(args.n, len(entrees)))

    chemin_rejets = Path(args.out).with_suffix(".rejets.jsonl")
    n_ok, n_rejet = 0, 0

    with open(args.out, "w", encoding="utf-8") as f_out, \
         open(chemin_rejets, "w", encoding="utf-8") as f_rejets:
        for entree in echantillon:
            dialogue, erreur = traduire_entree(entree, model=args.model)
            if dialogue is not None:
                f_out.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
                n_ok += 1
                print(f"[ok]    qid {entree['qid']}")
            else:
                f_rejets.write(json.dumps({"qid": entree["qid"], "raison": erreur}, ensure_ascii=False) + "\n")
                n_rejet += 1
                print(f"[rejet] qid {entree['qid']} -> {erreur}")

    attribution_path = Path(args.out).parent / "ATTRIBUTION.md"
    if not attribution_path.exists():
        attribution_path.write_text(ATTRIBUTION_TEXTE, encoding="utf-8")

    print(f"\nTerminé : {n_ok} dialogues traduits dans {args.out}, {n_rejet} rejets dans {chemin_rejets}")
    print(f"Pense à garder {attribution_path.name} avec ces données (licence CC BY-SA 4.0).")


if __name__ == "__main__":
    main()
