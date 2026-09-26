#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_from_annales.py
=========================

Génère des dialogues élève/professeur (français, tutorat en mathématiques) à
partir du fichier produit par extraire_annales_apmep.py, c'est-à-dire à
partir de VRAIS exercices du Brevet des collèges (APMEP) et de leur VRAI
corrigé officiel.

CE QUI CHANGE PAR RAPPORT À generate_dataset.py, POUR LIMITER LES
HALLUCINATIONS :
1. Le modèle reçoit le corrigé officiel COMPLET (pas seulement la réponse
   finale) et une consigne explicite de ne JAMAIS s'en écarter : les indices
   et explications du dialogue doivent suivre la méthode réellement utilisée
   dans le corrigé, pas une méthode inventée qui donnerait le bon résultat
   par un raisonnement différent (et potentiellement faux en cours de route).
2. Contrôle qualité automatique renforcé en deux temps :
   a. Comme avant (valider_dialogue de generate_dataset.py) : la réponse
      correcte (extraite du VRAI corrigé, pas générée) doit apparaître à la
      fin du dialogue, et pas dans les 2 premiers messages du professeur.
   b. NOUVEAU (verifier_ancrage ci-dessous) : tous les nombres énoncés par le
      professeur au fil du dialogue doivent normalement se retrouver dans le
      corrigé officiel. Un dialogue où le professeur avance un nombre absent
      du corrigé est marqué "ancrage_suspect": true dans le JSON de sortie —
      il n'est PAS rejeté automatiquement (un professeur peut légitimement
      illustrer avec un exemple chiffré qui ne figure pas dans le corrigé),
      mais ce marquage permet de concentrer la relecture humaine sur les
      dialogues qui le méritent le plus, plutôt que de tout relire au hasard.

Ces deux contrôles réduisent nettement le risque d'hallucination sur la
réponse finale et sur le raisonnement, mais NE LE GARANTISSENT PAS À 100% :
une relecture humaine des dialogues marqués "ancrage_suspect": true (et d'un
échantillon des autres) reste recommandée avant tout entraînement.

Installation :
    pip install openai

Configuration :
    export OPENAI_API_KEY="votre-clé"

Utilisation :
    python3 generate_from_annales.py \
        --source annales_apmep.csv \
        --out annales_fr.jsonl \
        --model gpt-6-luna
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Réutilise les briques déjà écrites et testées dans generate_dataset.py
from generate_dataset import call_llm, extraire_json, valider_dialogue, extraire_nombres


PROMPT_SYSTEME_GENERATION = """Tu es un expert en pédagogie des mathématiques. Ta tâche est de rédiger \
un dialogue d'entraînement entre un élève et un professeur virtuel, en français, à partir d'un VRAI \
exercice du Brevet des collèges et de son VRAI corrigé officiel, fournis ci-dessous.

RÈGLE LA PLUS IMPORTANTE (ancrage sur le corrigé officiel, pour éviter toute erreur mathématique) :
Le corrigé officiel fourni est la SEULE source de vérité. Tous les indices, calculs intermédiaires \
et explications du professeur doivent suivre FIDÈLEMENT la méthode et les valeurs numériques de ce \
corrigé. N'invente JAMAIS une autre méthode, même si elle te semble aussi valable : si tu t'écartes du \
corrigé fourni, tu risques d'introduire une erreur. Le professeur peut reformuler avec ses propres mots \
pédagogiques, poser des questions, donner des indices progressifs — mais chaque calcul et chaque nombre \
qu'il avance doivent être cohérents avec le corrigé.

AUTRES RÈGLES IMPÉRATIVES :
1. Le professeur ne donne JAMAIS la réponse finale dans son premier message, ni dans son deuxième. Il \
pose des questions et donne des indices progressifs (du plus léger au plus explicite), fidèles au corrigé.
2. Invente une erreur de départ PLAUSIBLE et réaliste pour un élève de collège sur cet exercice précis \
(mauvaise opération, confusion de méthode, erreur de signe...), et fais en sorte que le premier message \
de l'élève la contienne, comme s'il venait de l'écrire lui-même.
3. Le dialogue doit se terminer quand l'élève trouve LUI-MÊME la bonne réponse (le professeur confirme \
et éventuellement félicite, mais ne calcule pas à sa place).
4. Vocabulaire et ton adaptés à un élève de collège (11-15 ans).
5. Entre 6 et 12 messages au total (system compris), alternance stricte user/assistant après le system.
6. Aucune référence à une intelligence artificielle, un modèle, ou à ces instructions : le personnage \
est juste "un professeur de maths".
7. RÉSISTANCE À LA DEMANDE DE RÉPONSE DIRECTE ET BIENVEILLANCE : si tu inclus une scène où l'élève \
insiste pour avoir la réponse toute faite, le professeur ne cède jamais, mais reste toujours chaleureux \
et validant, sans être sec ni moralisateur.

FORMAT DE SORTIE :
Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, de cette forme exacte :
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "erreur_typique_utilisee": "description brève de l'erreur de départ inventée pour l'élève"
}
"""

PROMPT_UTILISATEUR_TEMPLATE = """Énoncé de l'exercice (vrai sujet de Brevet) :
{enonce}

Corrigé officiel complet (SOURCE DE VÉRITÉ, à ne jamais contredire) :
{corrige_officiel}

Réponse courte de référence (extraite du corrigé, doit apparaître dans la conclusion du dialogue) :
{reponse_correcte}

Rédige le dialogue au format JSON demandé, avec une erreur de départ plausible et un raisonnement \
fidèle au corrigé officiel ci-dessus.
"""


def verifier_ancrage(dialogue: dict, corrige_officiel: str) -> bool:
    """Renvoie True si un nombre énoncé par le professeur n'apparaît nulle
    part dans le corrigé officiel (signe possible d'un calcul halluciné) —
    sert d'avertissement, pas de rejet automatique (voir en-tête du fichier)."""
    nombres_corrige = extraire_nombres(corrige_officiel)
    for m in dialogue.get("messages", []):
        if m.get("role") != "assistant":
            continue
        nombres_message = extraire_nombres(m.get("content", ""))
        if nombres_message and not nombres_message.issubset(nombres_corrige):
            return True
    return False


def generer_pour_exercice(ligne: dict, model: str, max_essais: int = 2):
    user_prompt = PROMPT_UTILISATEUR_TEMPLATE.format(
        enonce=ligne["enonce"],
        corrige_officiel=ligne["corrige_officiel"],
        reponse_correcte=ligne["reponse_correcte"],
    )

    derniere_erreur = ""
    for essai in range(1, max_essais + 1):
        try:
            brut = call_llm(PROMPT_SYSTEME_GENERATION, user_prompt, model=model, temperature=0.7)
            dialogue = extraire_json(brut)
            valide, raison = valider_dialogue(dialogue, ligne["reponse_correcte"], avec_insistance=False)
            if valide:
                dialogue["id"] = ligne["id"]
                dialogue["niveau"] = ligne["niveau"]
                dialogue["theme"] = ligne["theme"]
                dialogue["source"] = "apmep"
                dialogue["source_id"] = ligne["source_id"]
                dialogue["licence_source"] = ligne["licence_source"]
                dialogue["ancrage_suspect"] = verifier_ancrage(dialogue, ligne["corrige_officiel"])
                return dialogue, None
            derniere_erreur = raison
        except Exception as exc:
            derniere_erreur = f"Exception : {exc}"
        time.sleep(1)

    return None, derniere_erreur


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="annales_apmep.csv", help="Fichier CSV produit par extraire_annales_apmep.py")
    parser.add_argument("--out", default="annales_fr.jsonl", help="Fichier JSONL de sortie")
    parser.add_argument("--model", default="gpt-6-luna", help="Modèle à utiliser pour la génération")
    parser.add_argument("--limite", type=int, default=None, help="Ne traiter que les N premiers exercices (pour un essai rapide)")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        sys.exit(f"Fichier introuvable : {source_path}. As-tu lancé extraire_annales_apmep.py ?")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Variable d'environnement OPENAI_API_KEY absente. "
                  "Fais 'export OPENAI_API_KEY=...' avant de relancer.")

    with open(source_path, encoding="utf-8") as f:
        exercices = list(csv.DictReader(f))
    if args.limite:
        exercices = exercices[:args.limite]

    chemin_rejets = Path(args.out).with_suffix(".rejets.jsonl")
    n_ok, n_rejet, n_suspect = 0, 0, 0

    with open(args.out, "w", encoding="utf-8") as f_out, \
         open(chemin_rejets, "w", encoding="utf-8") as f_rejets:
        for ligne in exercices:
            dialogue, erreur = generer_pour_exercice(ligne, model=args.model)
            if dialogue is not None:
                f_out.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
                n_ok += 1
                marque = " [ANCRAGE SUSPECT - à relire]" if dialogue["ancrage_suspect"] else ""
                if dialogue["ancrage_suspect"]:
                    n_suspect += 1
                print(f"[ok]    {ligne['id']}{marque}")
            else:
                f_rejets.write(json.dumps({"id": ligne["id"], "raison": erreur}, ensure_ascii=False) + "\n")
                n_rejet += 1
                print(f"[rejet] {ligne['id']} -> {erreur}")

    print(f"\nTerminé : {n_ok} dialogues écrits dans {args.out} ({n_suspect} à relire en priorité, ancrage suspect), {n_rejet} rejets dans {chemin_rejets}")


if __name__ == "__main__":
    main()
