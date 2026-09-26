#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_eedi.py
==================

Traduit et adapte en français un échantillon du dataset public Eedi
(Question-Anchored-Tutoring-Dialogues-2k) : de VRAIES conversations de
tutorat (humain-humain, pas générées), ancrées sur une question à choix
multiples (QCM) du programme britannique.

Différences structurelles avec MathDial / SocraTeach, à bien comprendre
avant d'utiliser ce script :

1. Ce sont de vraies conversations, pas des dialogues rédigés pour un
   papier de recherche : le ton est informel (salutations, digressions,
   plusieurs messages d'affilée de la même personne). Ce script fusionne
   d'abord les messages consécutifs d'un même interlocuteur en un seul
   tour avant de traduire.
2. Le problème est un QCM à 4 réponses (A/B/C/D), souvent formulé comme
   "Untel dit X, Unetelle dit Y, qui a raison ?", parfois avec du LaTeX
   ou une image (les images ne sont pas traduisibles ici : elles sont
   signalées mais pas décrites).
3. AUCUN champ des données d'origine n'indique directement la bonne
   réponse : le script demande donc au modèle de traduction de la
   déduire lui-même de la façon dont le tuteur conclut la conversation,
   et de l'indiquer dans un champ "lettre_correcte" du JSON produit.

   Conséquence importante sur le contrôle qualité automatique : pour
   MathDial/SocraTeach, la bonne réponse est un nombre connu à l'avance,
   donc le script peut vérifier avec certitude qu'elle apparaît bien à
   la fin du dialogue. Ici, une bonne partie des options Eedi sont du
   texte ("Only Alex", "Neither is correct"...) et pas des nombres : le
   contrôle automatique de contenu (valider_dialogue_qcm ci-dessous) ne
   peut vérifier fermement QUE les QCM à options numériques ; pour les
   options textuelles, il vérifie la structure du dialogue mais ne peut
   pas garantir que la bonne réponse est bien confirmée à la fin. LA
   RELECTURE HUMAINE EST DONC ENCORE PLUS IMPORTANTE ICI QUE POUR
   MATHDIAL/SOCRATEACH.

LICENCE : Eedi/Question-Anchored-Tutoring-Dialogues-2k est publié sous
licence Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0),
comme SocraTeach : usage NON COMMERCIAL uniquement. Voir
ATTRIBUTION_EEDI.md (généré à côté du fichier de sortie).

LIMITE PRATIQUE DE L'ÉCHANTILLON FOURNI : les fichiers
eedi-source/anchored-dialogues.csv et eedi-source/dq-question-metadata.csv
livrés avec ce script ne contiennent qu'un tout petit échantillon réel
(6 interventions), récupéré via le connecteur Hugging Face — ce dépôt est
bloqué à l'accès réseau direct (git/curl) dans l'environnement où ce
script a été écrit, et le connecteur ne peut transférer que de petits
volumes par appel. Pour traiter les ~2000 interventions du dataset
complet, télécharge toi-même les fichiers CSV depuis :
  https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k/resolve/main/anchored-dialogues.csv
  https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k/resolve/main/dq-question-metadata.csv
et relance ce script avec --dialogues et --metadata pointant vers ces
fichiers complets (colonnes identiques à l'échantillon fourni).

Installation :
    pip install openai

Configuration :
    export OPENAI_API_KEY="votre-clé"

Utilisation :
    python3 translate_eedi.py \
        --dialogues eedi-source/anchored-dialogues.csv \
        --metadata eedi-source/dq-question-metadata.csv \
        --out eedi_fr.jsonl \
        --model gpt-6-luna
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Réutilise les briques déjà écrites et testées dans generate_dataset.py
from generate_dataset import call_llm, extraire_json, extraire_nombres

LETTRES = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# 1. Lecture et reconstruction des données d'origine
# ---------------------------------------------------------------------------

def charger_questions(chemin_metadata) -> dict:
    """Regroupe les lignes de dq-question-metadata.csv par QuestionId_DQ et
    reconstruit l'énoncé + les 4 options, dans l'ordre du champ 'Sequence'."""
    par_question = defaultdict(list)
    with open(chemin_metadata, encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            par_question[ligne["QuestionId_DQ"]].append(ligne)

    questions = {}
    for qid, lignes in par_question.items():
        lignes.sort(key=lambda l: int(l["Sequence"]))
        morceaux_enonce = []
        options = {}
        a_une_image = False
        for l in lignes:
            label = l["Label"]
            if label == "Question Text":
                morceaux_enonce.append(l["Text"])
            elif label == "Question Image":
                a_une_image = True
            elif label.startswith("Answer") and label.endswith("Text"):
                lettre = label.split()[1]  # "Answer A Text" -> "A"
                options[lettre] = l["Text"]
            elif label.startswith("Answer") and label.endswith("Image"):
                lettre = label.split()[1]
                options[lettre] = "[image, non traduisible]"
        questions[qid] = {
            "enonce": " ".join(morceaux_enonce),
            "options": options,
            "a_une_image": a_une_image,
        }
    return questions


def charger_dialogues(chemin_dialogues) -> dict:
    """Regroupe les lignes de anchored-dialogues.csv par InterventionId,
    triées par MessageSequence, et fusionne les messages consécutifs du même
    interlocuteur (même valeur de IsTutor) en un seul tour."""
    par_intervention = defaultdict(list)
    with open(chemin_dialogues, encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            par_intervention[ligne["InterventionId"]].append(ligne)

    interventions = {}
    for iid, lignes in par_intervention.items():
        lignes.sort(key=lambda l: int(l["MessageSequence"]))
        qid = lignes[0]["QuestionId_DQ"]

        tours = []
        for l in lignes:
            est_tuteur = l["IsTutor"] == "1"
            texte = l["MessageString"].strip()
            if not texte:
                continue
            if tours and tours[-1]["est_tuteur"] == est_tuteur:
                tours[-1]["texte"] += "\n" + texte
            else:
                tours.append({"est_tuteur": est_tuteur, "texte": texte})

        if tours:
            interventions[iid] = {"question_id": qid, "tours": tours}
    return interventions


def construire_transcription(tours: list) -> str:
    lignes = []
    for tour in tours:
        qui = "Tutor" if tour["est_tuteur"] else "Student"
        lignes.append(f"{qui}: {tour['texte']}")
    return "\n".join(lignes)


def construire_enonce_qcm(question: dict) -> str:
    lignes = [question["enonce"], ""]
    for lettre in LETTRES:
        if lettre in question["options"]:
            lignes.append(f"{lettre}) {question['options'][lettre]}")
    if question["a_une_image"]:
        lignes.append("\n(Cette question comporte une image, non décrite ici.)")
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# 2. Prompt de traduction
# ---------------------------------------------------------------------------

PROMPT_SYSTEME_TRADUCTION = """Tu es un traducteur spécialisé qui adapte, de l'anglais vers le \
français, une VRAIE conversation de tutorat en mathématiques entre un tuteur et un \
élève de collège/lycée britannique, ancrée sur une question à choix multiples (QCM).

RÈGLES IMPÉRATIVES :
1. C'est une vraie conversation humaine, pas un dialogue scénarisé : elle peut \
contenir des salutations, du bavardage, des messages informels. Traduis fidèlement \
le fond pédagogique (ce que dit le tuteur pour guider l'élève, ce que répond \
l'élève) mais tu peux raccourcir ou supprimer les pures politesses hors-sujet \
(bonjour, au revoir, émoticônes) si elles n'apportent rien à l'échange pédagogique. \
Ne supprime et ne déplace en revanche AUCUN indice, AUCUNE relance, AUCUNE tentative \
de réponse de l'élève.
2. Remplace le prénom du VRAI élève (celui qui participe à la conversation) par un \
prénom courant en France, et garde-le cohérent sur tout le dialogue. En revanche, si \
le QCM lui-même met en scène des personnages fictifs pour poser la question (par \
exemple "Alex dit ceci, Sophie dit cela, qui a raison ?"), NE change PAS leurs noms : \
ce sont des noms de l'énoncé, pas celui de l'élève.
3. Le QCM (énoncé + options A/B/C/D) fourni plus bas doit être traduit fidèlement et \
former, avec une courte formulation de l'élève ("je pense que c'est peut-être ... mais \
je suis pas sûr", à inventer sobrement si le premier message de l'élève dans la \
transcription n'indique pas clairement son choix de départ), le premier message \
"user" du JSON final.
4. La transcription fournie alterne "Tutor:" et "Student:" : traduis-la fidèlement, \
dans l'ordre (Tutor -> assistant, Student -> user), en respectant la règle 1 ci-dessus.
5. Le premier message "system" doit présenter un professeur de maths bienveillant qui \
ne donne jamais la réponse directement.
6. IMPORTANT : à partir de la façon dont le tuteur conclut ou confirme la réponse dans \
la transcription, détermine quelle lettre (A, B, C ou D) est la bonne réponse au QCM, \
et indique-la dans le champ "lettre_correcte" du JSON de sortie. Si ce n'est vraiment \
pas clair, donne ta meilleure estimation plutôt que de laisser le champ vide.

FORMAT DE SORTIE :
Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour :
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "lettre_correcte": "A"
}
"""

PROMPT_UTILISATEUR_TEMPLATE = """QCM (anglais) :
{enonce_qcm}

Transcription brute de la conversation réelle (anglais, Tutor = tuteur, Student = élève) :
{transcription_brute}

Traduis et restructure ce dialogue en français au format JSON demandé.
"""


# ---------------------------------------------------------------------------
# 3. Contrôle qualité automatique (adapté au format QCM, voir la remarque en
#    haut de fichier sur ses limites par rapport à MathDial/SocraTeach)
# ---------------------------------------------------------------------------

def valider_dialogue_qcm(dialogue: dict, options: dict) -> tuple:
    """Renvoie (est_valide, raison_si_rejet). Vérifie systématiquement la
    structure du dialogue et la validité de la lettre annoncée. Ne peut
    vérifier le CONTENU de la réponse finale que lorsque l'option correcte
    est numérique (cf. remarque de licence/limites en haut de fichier) :
    pour une option textuelle ("Only Alex", "Neither is correct"...), ce
    contrôle est sauté faute de pouvoir comparer fiablement un texte anglais
    d'origine à sa traduction française."""
    messages = dialogue.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        return False, "Moins de 4 messages, ou champ 'messages' absent/mal formé"

    if messages[0].get("role") != "system":
        return False, "Le premier message n'est pas de rôle 'system'"

    attendu = "user"
    for m in messages[1:]:
        if m.get("role") != attendu:
            return False, f"Alternance rompue : attendu '{attendu}', trouvé '{m.get('role')}'"
        attendu = "assistant" if attendu == "user" else "user"
    if messages[-1]["role"] != "assistant":
        return False, "Le dialogue ne se termine pas par le professeur"

    lettre = str(dialogue.get("lettre_correcte", "")).strip().upper()
    if lettre not in options:
        return False, f"Lettre correcte absente ou invalide : {lettre!r}"

    nombres_option = extraire_nombres(options[lettre])

    def contient_la_reponse(texte: str):
        """True/False si vérifiable (option numérique), None si l'option est
        textuelle et donc non vérifiable automatiquement ici."""
        if nombres_option:
            return nombres_option.issubset(extraire_nombres(texte))
        return None

    # messages[2] est le premier message "assistant" (0=system, 1=user, 2=assistant) ;
    # la structure a déjà été validée ci-dessus (>= 4 messages, alternance stricte).
    verif_premier = contient_la_reponse(messages[2]["content"])
    if verif_premier is True:
        return False, "Le professeur donne la réponse dès le premier message"

    verif_dernier = contient_la_reponse(messages[-1]["content"])
    if verif_dernier is False:
        return False, "La réponse numérique correcte n'apparaît pas dans la conclusion"
    # verif_dernier is None -> option textuelle, non vérifiable automatiquement : on laisse passer

    return True, ""


# ---------------------------------------------------------------------------
# 4. Boucle principale
# ---------------------------------------------------------------------------

def traduire_entree(iid: str, intervention: dict, questions: dict, model: str, max_essais: int = 2):
    qid = intervention["question_id"]
    question = questions.get(qid)
    if question is None:
        return None, f"Question {qid} absente du fichier de métadonnées"

    user_prompt = PROMPT_UTILISATEUR_TEMPLATE.format(
        enonce_qcm=construire_enonce_qcm(question),
        transcription_brute=construire_transcription(intervention["tours"]),
    )

    derniere_erreur = ""
    for essai in range(1, max_essais + 1):
        try:
            brut = call_llm(PROMPT_SYSTEME_TRADUCTION, user_prompt, model=model, temperature=0.3)
            dialogue = extraire_json(brut)
            valide, raison = valider_dialogue_qcm(dialogue, question["options"])
            if valide:
                dialogue["id"] = f"eedi-fr-{iid}"
                dialogue["niveau"] = "college"  # programme UK, à ajuster au cas par cas
                dialogue["theme"] = "qcm_raisonnement"
                dialogue["source"] = "eedi"
                dialogue["source_id"] = iid
                dialogue["question_id_origine"] = qid
                dialogue["licence_source"] = "CC BY-NC 4.0 (non commercial)"
                return dialogue, None
            derniere_erreur = raison
        except Exception as exc:
            derniere_erreur = f"Exception : {exc}"
        time.sleep(1)

    return None, derniere_erreur


ATTRIBUTION_TEXTE = """# Attribution requise (dataset Eedi)

Les dialogues marqués `"source": "eedi"` dans ce jeu de données sont \
traduits et adaptés à partir de Eedi/Question-Anchored-Tutoring-Dialogues-2k, \
publié sous licence Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
https://creativecommons.org/licenses/by-nc/4.0/

**Usage NON COMMERCIAL uniquement** — comme SocraTeach, plus restrictif que \
MathDial (CC BY-SA). Si tu envisages un jour de distribuer commercialement le \
modèle fine-tuné ou l'app qui l'utilise, il faudra exclure ces dialogues du jeu \
d'entraînement (filtre facile : `"source" != "eedi"`).

Dépôt d'origine (Hugging Face) :
https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k

Rappel important : contrairement à MathDial/SocraTeach, la bonne réponse n'est \
pas garantie par un contrôle automatique fiable pour les QCM à options \
textuelles (voir l'en-tête de translate_eedi.py) — relis à la main un \
échantillon plus large que d'habitude avant d'utiliser ces dialogues pour \
l'entraînement.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dialogues", default="eedi-source/anchored-dialogues.csv", help="Fichier CSV des dialogues Eedi d'origine")
    parser.add_argument("--metadata", default="eedi-source/dq-question-metadata.csv", help="Fichier CSV des métadonnées de questions Eedi")
    parser.add_argument("--out", default="eedi_fr.jsonl", help="Fichier JSONL de sortie (français)")
    parser.add_argument("--n", type=int, default=None, help="Nombre d'interventions à traiter (par défaut : toutes celles trouvées dans --dialogues)")
    parser.add_argument("--model", default="gpt-6-luna", help="Modèle à utiliser pour la traduction")
    args = parser.parse_args()

    chemin_dialogues = Path(args.dialogues)
    chemin_metadata = Path(args.metadata)
    if not chemin_dialogues.exists():
        sys.exit(f"Fichier introuvable : {chemin_dialogues}")
    if not chemin_metadata.exists():
        sys.exit(f"Fichier introuvable : {chemin_metadata}")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Variable d'environnement OPENAI_API_KEY absente. "
                  "Fais 'export OPENAI_API_KEY=...' avant de relancer.")

    questions = charger_questions(chemin_metadata)
    interventions = charger_dialogues(chemin_dialogues)

    ids = list(interventions.keys())
    if args.n:
        ids = ids[:args.n]

    chemin_rejets = Path(args.out).with_suffix(".rejets.jsonl")
    n_ok, n_rejet = 0, 0

    with open(args.out, "w", encoding="utf-8") as f_out, \
         open(chemin_rejets, "w", encoding="utf-8") as f_rejets:
        for iid in ids:
            dialogue, erreur = traduire_entree(iid, interventions[iid], questions, model=args.model)
            if dialogue is not None:
                f_out.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
                n_ok += 1
                print(f"[ok]    intervention {iid}")
            else:
                f_rejets.write(json.dumps({"id": iid, "raison": erreur}, ensure_ascii=False) + "\n")
                n_rejet += 1
                print(f"[rejet] intervention {iid} -> {erreur}")

    attribution_path = Path(args.out).parent / "ATTRIBUTION_EEDI.md"
    if not attribution_path.exists():
        attribution_path.write_text(ATTRIBUTION_TEXTE, encoding="utf-8")

    print(f"\nTerminé : {n_ok} dialogues traduits dans {args.out}, {n_rejet} rejets dans {chemin_rejets}")
    print(f"Pense à garder {attribution_path.name} avec ces données (licence CC BY-NC 4.0 — non commercial).")
    print("Rappel : relecture humaine encore plus importante ici que pour MathDial/SocraTeach (voir en-tête du script).")


if __name__ == "__main__":
    main()
