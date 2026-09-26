#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_socrateach.py
=========================

Traduit et adapte en français un échantillon du dataset public SocraTeach
(SocraticLM, NeurIPS 2024) : dialogues de tutorat en mathématiques,
construits sur les mêmes problèmes que MathDial (GSM8K, niveau collège),
mais avec plusieurs variantes de dialogue par problème et des étiquettes
d'état cognitif de l'élève ("user_type").

ATTENTION LICENCE : contrairement à MathDial (CC BY-SA, réutilisation
commerciale possible avec partage à l'identique), SocraTeach est publié
sous licence Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
— voir socraticlm-source/LICENSE/DATA_LICENSE. Tout dialogue traduit
d'après cette source (champ "source": "socrateach") ne doit être utilisé
qu'à des fins NON COMMERCIALES. Si le modèle fine-tuné ou l'app qui
l'utilise devait un jour être vendu ou monétisé, il faudrait exclure ces
dialogues (ou obtenir une autre autorisation des auteurs).

Comme MathDial, ce script fait une TRADUCTION FIDÈLE (il ne change pas
le déroulé pédagogique). Le renforcement "résistance à la demande de
réponse directe" reste dans generate_dataset.py, sur des dialogues créés
depuis zéro.

Installation :
    pip install openai

Configuration :
    export OPENAI_API_KEY="votre-clé"

Utilisation :
    python3 translate_socrateach.py \
        --source socraticlm-source/data/SocraTeach_multi.json \
        --out socrateach_fr.jsonl \
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

from generate_dataset import call_llm, extraire_json, valider_dialogue


PROMPT_SYSTEME_TRADUCTION = """Tu es un traducteur spécialisé qui adapte un dialogue de tutorat en \
mathématiques, de l'anglais vers le français, pour un usage avec des élèves français de collège.

RÈGLES IMPÉRATIVES :
1. NE MODIFIE PAS le déroulé pédagogique du dialogue d'origine : traduis \
fidèlement chaque relance du professeur et chaque réponse de l'élève, dans \
le même ordre. Tu ne modifies pas, tu n'ajoutes pas de nouveau contenu \
pédagogique et tu n'en retires pas.
2. Dans les données d'origine, l'élève n'a pas encore tenté de résoudre le \
problème : le premier message "user" du dialogue final doit donc présenter \
l'énoncé traduit du problème, suivi d'une courte phrase où l'élève indique \
qu'il ne sait pas trop comment commencer (à formuler naturellement, par \
exemple "je ne sais pas trop par où commencer" ou équivalent).
3. La transcription brute fournie plus bas alterne "Teacher:" (le professeur) \
et "Student:" (l'élève) ligne par ligne, dans l'ordre : traduis-la fidèlement \
en conservant qui parle (Teacher -> assistant, Student -> user).
4. Le tout dernier message de la transcription (marqué par une fin de \
dialogue) doit devenir le dernier message "assistant" du JSON final.
5. Le premier message "system" doit présenter un professeur de maths \
bienveillant qui ne donne jamais la réponse directement.

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

Réponse numérique correcte attendue : {reponse_correcte}

Transcription brute du dialogue (anglais, à traduire tour par tour dans l'ordre) :
{transcription_brute}

Traduis et restructure ce dialogue en français au format JSON demandé.
"""


def construire_transcription(tours: list) -> str:
    """Reconstruit une transcription texte lisible à partir de la liste de
    tours SocraTeach (chaque tour a une clé 'system' = professeur, et
    éventuellement 'user' = élève ; le dernier tour n'a pas de 'user')."""
    lignes = []
    for tour in tours:
        lignes.append(f"Teacher: {tour['system']}")
        if "user" in tour:
            lignes.append(f"Student: {tour['user']}")
    return "\n".join(lignes)


def traduire_entree(cle_probleme: str, cle_variante: str, entree: dict, tours: list, model: str, max_essais: int = 2):
    reponse_correcte = str(entree["answer"])
    user_prompt = PROMPT_UTILISATEUR_TEMPLATE.format(
        question=entree["question"],
        reponse_correcte=reponse_correcte,
        transcription_brute=construire_transcription(tours),
    )

    derniere_erreur = ""
    for essai in range(1, max_essais + 1):
        try:
            brut = call_llm(PROMPT_SYSTEME_TRADUCTION, user_prompt, model=model, temperature=0.3)
            dialogue = extraire_json(brut)
            valide, raison = valider_dialogue(dialogue, reponse_correcte, avec_insistance=False)
            if valide:
                dialogue["id"] = f"socrateach-fr-{cle_variante}"
                dialogue["niveau"] = "college"  # GSM8K ~ 5e française, comme MathDial
                dialogue["theme"] = "probleme_multi_etapes"
                dialogue["source"] = "socrateach"
                dialogue["source_id"] = cle_variante
                dialogue["licence_source"] = "CC BY-NC 4.0 (non commercial)"
                return dialogue, None
            derniere_erreur = raison
        except Exception as exc:
            derniere_erreur = f"Exception : {exc}"
        time.sleep(1)

    return None, derniere_erreur


ATTRIBUTION_TEXTE = """# Attribution requise (dataset SocraTeach / SocraticLM)

Les dialogues marqués `"source": "socrateach"` dans ce jeu de données sont \
traduits et adaptés à partir de SocraTeach, publié sous licence \
Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
https://creativecommons.org/licenses/by-nc/4.0/

**Usage NON COMMERCIAL uniquement** — c'est plus restrictif que MathDial \
(CC BY-SA, qui autorise un usage commercial avec partage à l'identique). \
Si tu envisages un jour de distribuer commercialement le modèle fine-tuné \
ou l'app qui l'utilise, il faudra exclure ces dialogues du jeu \
d'entraînement (filtre facile : `"source" != "socrateach"`).

Citation à conserver :
Liu, Jiayu, Huang, Zhenya, Xiao, Tong, Sha, Jing, Wu, Jinze, Liu, Qi, Wang, \
Shijin, and Chen, Enhong. 2024. SocraticLM: Exploring Socratic Personalized \
Teaching with Large Language Models. Advances in Neural Information \
Processing Systems, 37, 85693-85721.

Dépôt d'origine : https://github.com/Ljyustc/SocraticLM
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="socraticlm-source/data/SocraTeach_multi.json", help="Fichier SocraTeach d'origine (.json)")
    parser.add_argument("--out", default="socrateach_fr.jsonl", help="Fichier JSONL de sortie (français)")
    parser.add_argument("--n", type=int, default=50, help="Nombre de problèmes à échantillonner (1 variante traduite par problème)")
    parser.add_argument("--model", default="gpt-6-luna", help="Modèle à utiliser pour la traduction")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour l'échantillonnage (reproductibilité)")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        sys.exit(f"Fichier introuvable : {source_path}. As-tu cloné Ljyustc/SocraticLM ?")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Variable d'environnement OPENAI_API_KEY absente. "
                  "Fais 'export OPENAI_API_KEY=...' avant de relancer.")

    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    cles_problemes = random.sample(list(data.keys()), min(args.n, len(data)))

    chemin_rejets = Path(args.out).with_suffix(".rejets.jsonl")
    n_ok, n_rejet = 0, 0

    with open(args.out, "w", encoding="utf-8") as f_out, \
         open(chemin_rejets, "w", encoding="utf-8") as f_rejets:
        for cle_probleme in cles_problemes:
            entree = data[cle_probleme]
            # Une seule variante par problème (la première disponible), pour
            # avoir de la diversité de problèmes plutôt que plusieurs versions
            # du même. Change vers plusieurs variantes si tu veux plus de volume.
            cle_variante = next(iter(entree["dialogues"]))
            tours = entree["dialogues"][cle_variante]

            dialogue, erreur = traduire_entree(cle_probleme, cle_variante, entree, tours, model=args.model)
            if dialogue is not None:
                f_out.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
                n_ok += 1
                print(f"[ok]    {cle_variante}")
            else:
                f_rejets.write(json.dumps({"id": cle_variante, "raison": erreur}, ensure_ascii=False) + "\n")
                n_rejet += 1
                print(f"[rejet] {cle_variante} -> {erreur}")

    attribution_path = Path(args.out).parent / "ATTRIBUTION_SOCRATEACH.md"
    if not attribution_path.exists():
        attribution_path.write_text(ATTRIBUTION_TEXTE, encoding="utf-8")

    print(f"\nTerminé : {n_ok} dialogues traduits dans {args.out}, {n_rejet} rejets dans {chemin_rejets}")
    print(f"Pense à garder {attribution_path.name} avec ces données (licence CC BY-NC 4.0 — non commercial).")


if __name__ == "__main__":
    main()
