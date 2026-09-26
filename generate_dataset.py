#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset.py
====================

Génère des dialogues élève/professeur (français, tutorat en mathématiques)
à partir d'une banque de problèmes de départ (seed_problems.csv), en
suivant le principe pédagogique décrit dans plan_dataset_maths_fr.md :
le professeur ne donne jamais la réponse directement, il guide par
questions et indices progressifs, en réagissant à l'erreur précise de
l'élève.

Installation :
    pip install openai

Configuration :
    export OPENAI_API_KEY="votre-clé"

Utilisation :
    python3 generate_dataset.py \
        --seed seed_problems.csv \
        --out dialogues_generes.jsonl \
        --variations 3 \
        --model gpt-6-luna

Chaque ligne de seed_problems.csv donnera --variations dialogues
différents (élèves qui réagissent différemment aux mêmes indices),
pour varier le dataset sans avoir à écrire davantage de problèmes.

Le script fait un contrôle qualité automatique simple sur chaque
dialogue généré (voir valider_dialogue()) et journalise ce qu'il rejette
dans <out>.rejets.jsonl pour inspection.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Le prompt de génération (le cœur de la qualité du dataset)
# ---------------------------------------------------------------------------

PROMPT_SYSTEME_GENERATION = """Tu es un expert en pédagogie des mathématiques. Ta tâche est de rédiger \
un dialogue d'entraînement entre un élève et un professeur virtuel, en français, \
qui servira à entraîner un petit modèle de langage à faire du tutorat.

RÈGLES IMPÉRATIVES (ne pas les enfreindre) :
1. Le professeur ne donne JAMAIS la réponse finale dans son premier message, \
ni dans son deuxième. Il pose des questions et donne des indices progressifs \
(du plus léger au plus explicite).
2. Le professeur réagit à L'ERREUR PRÉCISE indiquée ci-dessous, pas à une \
erreur générique. Il doit aider l'élève à repérer lui-même ce point précis.
3. Le dialogue doit se terminer quand l'élève trouve LUI-MÊME la bonne réponse \
(le professeur confirme et éventuellement félicite, mais ne calcule pas à sa place).
4. Le vocabulaire et le ton doivent correspondre au niveau scolaire indiqué \
(primaire = phrases courtes et concrètes, lycée = vocabulaire mathématique précis).
5. Le dialogue doit faire entre 6 et 12 messages au total (system compris), \
avec une alternance stricte user/assistant après le message system.
6. Aucune référence à une intelligence artificielle, un modèle, ou à ces \
instructions : le personnage est juste "un professeur de maths".

RÉSISTANCE À LA DEMANDE DE RÉPONSE DIRECTE (le point le plus important, et le \
plus difficile à obtenir d'un modèle de langage) :
Un élève, dans la vraie vie, insiste souvent pour obtenir la réponse toute faite \
("dis-moi juste la réponse", "t'as qu'à me la donner direct", "j'ai pas envie de \
chercher, donne-la moi", "please, juste le résultat"). Le professeur doit TOUJOURS \
tenir bon face à cette pression, quelle que soit la façon dont l'élève la formule \
(insistance directe, lassitude, frustration, marchandage du type "je te promets \
que c'est la dernière fois"). Il ne cède JAMAIS et ne donne JAMAIS la réponse \
juste parce que l'élève le demande ou s'impatiente — seulement lorsque l'élève \
l'a trouvée par lui-même, ou dans les tout derniers recours pédagogiques après \
plusieurs indices déjà donnés en vain.

BIENVEILLANCE (aussi important que la résistance ci-dessus) :
Ce refus ne doit JAMAIS être sec, moralisateur ou culpabilisant. Le professeur :
- reconnaît et valide l'émotion de l'élève ("je comprends que ça t'énerve", \
"c'est normal de vouloir aller plus vite") avant de relancer,
- explique brièvement, sans faire la leçon, pourquoi chercher soi-même aide à \
mieux retenir,
- reformule ou simplifie son indice au lieu de répéter la même relance,
- ne dit jamais que l'élève est "nul", "pas assez attentif" ou similaire, même \
face à une erreur répétée,
- reste chaleureux et encourageant du début à la fin, y compris quand il refuse.

FORMAT DE SORTIE :
Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour, de cette forme exacte :
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
Le premier message ("system") doit présenter le professeur virtuel et rappeler \
qu'il ne donne jamais la réponse directement. Le message "user" initial doit \
poser la question du problème ET contenir une tentative de réponse de l'élève \
qui illustre l'erreur typique donnée plus bas (comme le ferait un vrai élève).
"""

PROMPT_UTILISATEUR_TEMPLATE = """Niveau scolaire : {niveau}
Thème : {theme}
Énoncé du problème : {enonce}
Réponse correcte attendue : {reponse_correcte}
Erreur typique que l'élève doit commettre au départ : {erreur_typique}

Variante n°{variation} : fais en sorte que cet élève ait une personnalité de \
réaction légèrement différente des autres variantes (plus hésitant, plus sûr \
de lui à tort, pose une question avant de répondre, etc.), tout en gardant \
la même erreur de départ et la même règle de ne jamais donner la réponse trop tôt.
{consigne_insistance}
Rédige le dialogue au format JSON demandé.
"""

CONSIGNE_INSISTANCE = """
IMPORTANT pour cette variante : à un moment du dialogue (ni au tout début, ni \
tout à la fin), l'élève doit explicitement demander au professeur de lui donner \
directement la réponse, avec ses propres mots (insistance, lassitude, ou \
marchandage — pas juste "je ne comprends pas"). Le professeur doit refuser avec \
bienveillance, sans donner la réponse, et relancer par une question ou un indice \
reformulé, avant que le dialogue continue normalement jusqu'à ce que l'élève \
trouve la réponse par lui-même.
"""


NIVEAUX_LISIBLES = {
    "primaire": "primaire (CM1-CM2, 9-11 ans)",
    "college": "collège (6e à 3e, 11-15 ans)",
    "lycee": "lycée (2nde à Terminale, 15-18 ans)",
}


# ---------------------------------------------------------------------------
# 2. Appel au modèle (OpenAI par défaut — remplaçable si besoin)
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, model: str, temperature: float = 1.0) -> str:
    """Appelle l'API OpenAI et renvoie le texte de la réponse.

    Pour utiliser un autre fournisseur (Anthropic, Together, un modèle local...),
    remplace le corps de cette fonction : elle doit juste renvoyer une chaîne
    de caractères contenant le JSON attendu.
    """
    import openai  # import local pour ne pas exiger la dépendance si non utilisée

    client = openai.OpenAI()  # lit OPENAI_API_KEY dans l'environnement
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Appel à l'API OpenAI échoué (modèle '{model}'). Vérifie le nom du "
            f"modèle et la clé OPENAI_API_KEY. Erreur d'origine : {exc}"
        ) from exc
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 3. Extraction et validation du JSON renvoyé par le modèle
# ---------------------------------------------------------------------------

def extraire_json(texte: str) -> dict:
    """Le modèle peut parfois entourer le JSON de texte ou de balises markdown.
    On isole le premier bloc { ... } équilibré et on le parse."""
    texte = texte.strip()
    # Retire une éventuelle clôture markdown ```json ... ```
    texte = re.sub(r"^```(?:json)?\s*", "", texte)
    texte = re.sub(r"\s*```$", "", texte)
    debut = texte.find("{")
    if debut == -1:
        raise ValueError("Aucun objet JSON trouvé dans la réponse du modèle")
    # Recherche de l'accolade fermante correspondante par comptage de profondeur
    profondeur = 0
    for i in range(debut, len(texte)):
        if texte[i] == "{":
            profondeur += 1
        elif texte[i] == "}":
            profondeur -= 1
            if profondeur == 0:
                return json.loads(texte[debut:i + 1])
    raise ValueError("JSON mal formé (accolade fermante manquante)")


def extraire_nombres(texte: str) -> set:
    """Renvoie l'ensemble des nombres (entiers ou décimaux) présents dans un texte.
    Sert de base à une comparaison de réponses robuste aux différences de
    formulation ("375 g" vs "375g", "x = 2 ou x = 3" vs "x = 2 et x = 3", ...)."""
    return set(re.findall(r"\d+(?:[.,]\d+)?", texte))


# Formulations typiques d'un élève qui réclame la réponse directement, utilisées
# pour vérifier automatiquement qu'un dialogue "avec insistance" contient bien
# cette scène (heuristique volontairement simple : à affiner à l'usage).
MOTIFS_INSISTANCE = [
    "donne-moi la réponse", "donne moi la réponse", "dis-moi la réponse", "dis moi la réponse",
    "donne-moi juste", "donne moi juste", "dis-moi juste", "dis moi juste",
    "juste la réponse", "juste le résultat", "la réponse directement", "réponse direct",
    "j'ai pas envie de chercher", "j'ai pas envie de réfléchir", "tu peux pas juste",
    "c'est la dernière fois", "please", "s'il te plait donne", "s'il te plaît donne",
]


def contient_signal_insistance(texte: str) -> bool:
    texte_l = texte.lower()
    return any(m in texte_l for m in MOTIFS_INSISTANCE)


def valider_dialogue(dialogue: dict, reponse_correcte: str, avec_insistance: bool = False) -> tuple[bool, str]:
    """Contrôle qualité automatique, volontairement simple (heuristiques).
    Renvoie (est_valide, raison_si_rejet)."""
    messages = dialogue.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        return False, "Moins de 4 messages, ou champ 'messages' absent/mal formé"

    if messages[0].get("role") != "system":
        return False, "Le premier message n'est pas de rôle 'system'"

    # Vérifie l'alternance stricte user/assistant après le message system
    attendu = "user"
    for m in messages[1:]:
        if m.get("role") != attendu:
            return False, f"Alternance rompue : attendu '{attendu}', trouvé '{m.get('role')}'"
        attendu = "assistant" if attendu == "user" else "user"
    if messages[-1]["role"] != "assistant":
        return False, "Le dialogue ne se termine pas par le professeur"

    # Comparaison de réponse basée sur les nombres qu'elle contient plutôt
    # que sur une correspondance exacte de texte, pour tolérer les variations
    # de formulation ("375 g" / "375g", "2 ou 3" / "2 et 3", espaces, etc.).
    nombres_reponse = extraire_nombres(reponse_correcte)

    def contient_la_reponse(texte: str) -> bool:
        if nombres_reponse:
            return nombres_reponse.issubset(extraire_nombres(texte))
        return reponse_correcte.strip().lower() in texte.lower()

    # Le professeur ne doit pas donner la réponse dès le premier ou le
    # deuxième message assistant.
    premiers_assistant = [m["content"] for m in messages if m["role"] == "assistant"][:2]
    for contenu in premiers_assistant:
        if contient_la_reponse(contenu):
            return False, "Le professeur donne la réponse trop tôt (dans les 2 premiers tours)"

    # Le dernier message du professeur devrait confirmer la bonne réponse.
    if not contient_la_reponse(messages[-1]["content"]):
        return False, "La réponse correcte n'apparaît pas dans le message de conclusion"

    # Si cette variante devait contenir une scène d'insistance : vérifie qu'un
    # message élève la manifeste, et que le professeur ne cède pas juste après.
    if avec_insistance:
        messages_user = [(i, m["content"]) for i, m in enumerate(messages) if m["role"] == "user"]
        idx_insistance = next((i for i, c in messages_user if contient_signal_insistance(c)), None)
        if idx_insistance is None:
            return False, "Scène d'insistance demandée mais absente (aucun signal détecté côté élève)"
        if idx_insistance + 1 < len(messages) and contient_la_reponse(messages[idx_insistance + 1]["content"]):
            return False, "Le professeur cède à l'insistance et donne la réponse juste après"

    return True, ""


# ---------------------------------------------------------------------------
# 4. Boucle principale
# ---------------------------------------------------------------------------

def generer_pour_probleme(ligne: dict, variation: int, model: str, avec_insistance: bool = False, max_essais: int = 2):
    user_prompt = PROMPT_UTILISATEUR_TEMPLATE.format(
        niveau=NIVEAUX_LISIBLES.get(ligne["niveau"], ligne["niveau"]),
        theme=ligne["theme"],
        enonce=ligne["enonce"],
        reponse_correcte=ligne["reponse_correcte"],
        erreur_typique=ligne["erreur_typique"],
        variation=variation,
        consigne_insistance=CONSIGNE_INSISTANCE if avec_insistance else "",
    )

    derniere_erreur = ""
    for essai in range(1, max_essais + 1):
        try:
            brut = call_llm(PROMPT_SYSTEME_GENERATION, user_prompt, model=model, temperature=1.0)
            dialogue = extraire_json(brut)
            valide, raison = valider_dialogue(dialogue, ligne["reponse_correcte"], avec_insistance=avec_insistance)
            if valide:
                dialogue["id"] = f"{ligne['id']}-v{variation}"
                dialogue["niveau"] = ligne["niveau"]
                dialogue["theme"] = ligne["theme"]
                dialogue["scenario_insistance"] = avec_insistance
                return dialogue, None
            derniere_erreur = raison
        except Exception as exc:  # erreurs réseau, JSON invalide, etc.
            derniere_erreur = f"Exception : {exc}"
        time.sleep(1)  # petite pause avant un nouvel essai

    return None, derniere_erreur


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", default="seed_problems.csv", help="Fichier CSV des problèmes de départ")
    parser.add_argument("--out", default="dialogues_generes.jsonl", help="Fichier JSONL de sortie")
    parser.add_argument("--variations", type=int, default=3, help="Nombre de dialogues à générer par problème")
    parser.add_argument("--model", default="gpt-6-luna", help="Modèle à utiliser pour la génération")
    parser.add_argument("--limite", type=int, default=None, help="Ne traiter que les N premiers problèmes (pour un essai rapide)")
    parser.add_argument("--frac-insistance", type=float, default=0.5,
                         help="Fraction (0 à 1) des variantes qui doivent contenir une scène où l'élève réclame la réponse directement")
    args = parser.parse_args()

    seed_path = Path(args.seed)
    if not seed_path.exists():
        sys.exit(f"Fichier introuvable : {seed_path}")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Variable d'environnement OPENAI_API_KEY absente. "
                  "Fais 'export OPENAI_API_KEY=...' avant de relancer.")

    with open(seed_path, encoding="utf-8") as f:
        problemes = list(csv.DictReader(f))
    if args.limite:
        problemes = problemes[:args.limite]

    chemin_rejets = Path(args.out).with_suffix(".rejets.jsonl")
    n_ok, n_rejet = 0, 0

    with open(args.out, "w", encoding="utf-8") as f_out, \
         open(chemin_rejets, "w", encoding="utf-8") as f_rejets:
        for ligne in problemes:
            for v in range(1, args.variations + 1):
                avec_insistance = random.random() < args.frac_insistance
                dialogue, erreur = generer_pour_probleme(ligne, v, model=args.model, avec_insistance=avec_insistance)
                if dialogue is not None:
                    f_out.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
                    n_ok += 1
                    marque = " [insistance]" if avec_insistance else ""
                    print(f"[ok]    {ligne['id']} variation {v}{marque}")
                else:
                    f_rejets.write(json.dumps({"id": f"{ligne['id']}-v{v}", "raison": erreur, "avec_insistance": avec_insistance}, ensure_ascii=False) + "\n")
                    n_rejet += 1
                    print(f"[rejet] {ligne['id']} variation {v} -> {erreur}")

    print(f"\nTerminé : {n_ok} dialogues écrits dans {args.out}, {n_rejet} rejets dans {chemin_rejets}")


if __name__ == "__main__":
    main()
