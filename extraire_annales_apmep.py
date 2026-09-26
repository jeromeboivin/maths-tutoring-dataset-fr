#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraire_annales_apmep.py
==========================

Construit un fichier "seed" (même esprit que seed_problems.csv) à partir de
vraies annales du Brevet des collèges publiées par l'APMEP
(https://www.apmep.fr/Annales-du-Brevet-des-colleges), sujet et corrigé étant
deux fichiers séparés, en PDF **ou en LaTeX (.tex)**.

POURQUOI CE SCRIPT EXISTE : l'objectif de Jérôme est un dataset dont les
réponses ne contiennent AUCUNE hallucination. La réponse correcte de chaque
exercice n'est donc pas inventée ni calculée par un modèle : elle est extraite
directement du texte du corrigé officiel. Le corrigé complet est aussi
conservé (colonne "corrige_officiel") pour servir de texte de référence
("grounding") au modèle qui rédigera ensuite le dialogue pédagogique, dans
generate_from_annales.py.

POURQUOI .TEX PLUTÔT QUE PDF QUAND C'EST POSSIBLE : le texte extrait d'un PDF
est souvent abîmé (colonnes mal remises dans l'ordre, formules éclatées). Le
LaTeX source, lui, est fiable et les formules restent intactes. Ce script
préfère donc le `.tex` quand il est disponible, et garde le PDF en repli pour
les années où seul le PDF existe. ATTENTION : les fichiers `.tex` de l'APMEP
sont tapuscrits par des bénévoles différents selon les années/académies, et
leurs conventions de mise en forme varient (voir plus bas pour le détail des
heuristiques et leurs limites).

CONTRAINTE RÉSEAU : ce script tourne dans un environnement qui n'a pas accès
à apmep.fr (site bloqué en sortie réseau ici, et son robots.txt renvoyait une
erreur 503 au moment où ce script a été écrit). Il faut donc TÉLÉCHARGER les
fichiers toi-même (depuis un navigateur normal, ça fonctionne) et les déposer
dans un dossier, avec une convention de nom simple pour que le script puisse
associer chaque sujet à son corrigé sans ambiguïté :

    <identifiant_session>__sujet.tex   (ou .pdf)
    <identifiant_session>__corrige.tex (ou .pdf)

Exemple :
    annales-source/2026-06-amerique-nord__sujet.tex
    annales-source/2026-06-amerique-nord__corrige.tex

(les noms de fichiers d'origine sur apmep.fr ne suivent pas une convention
assez régulière pour être associés automatiquement de façon fiable)

COMMENT LA RÉPONSE CORRECTE EST EXTRAITE (par ordre de préférence) :
1. Le dernier `\\fbox{...}` du corrigé de l'exercice, s'il y en a un — les
   correcteurs qui utilisent \\fbox l'utilisent presque toujours pour encadrer
   LA réponse à retenir, donc c'est le signal le plus fiable quand il existe.
2. À défaut (certains correcteurs ne l'utilisent jamais, et mettent la
   réponse en \\textbf{...} ou en texte simple) : le dernier "= <nombre>" du
   corrigé, puis à défaut le dernier nombre du texte, puis en tout dernier
   recours la dernière phrase telle quelle.

LIMITES IMPORTANTES (contrôle qualité) :
- Sur un exercice à plusieurs sous-questions indépendantes, cette réponse
  extraite ne couvre en général que la conclusion de la DERNIÈRE
  sous-question — les autres ne sont pas vérifiées automatiquement.
- Les conventions de mise en forme changent d'un tapuscripteur à l'autre :
  ce script vise à être robuste aux variantes vues jusqu'ici, mais une
  relecture humaine d'un échantillon reste recommandée, surtout au début.

Installation :
    pip install pdfplumber   (uniquement nécessaire si tu fournis des .pdf)

Utilisation :
    python3 extraire_annales_apmep.py --source annales-source --out annales_apmep.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Extraction du texte source (PDF ou LaTeX)
# ---------------------------------------------------------------------------

def extraire_texte_pdf(chemin: Path) -> str:
    import pdfplumber

    morceaux = []
    with pdfplumber.open(chemin) as pdf:
        for page in pdf.pages:
            morceaux.append(page.extract_text() or "")
    return "\n".join(morceaux)


COMMANDES_MISE_EN_PAGE = (r"\needspace", r"\vspace", r"\hspace", r"\addvspace")


def retirer_commandes_mise_en_page(texte: str) -> str:
    """Retire des commandes comme \\needspace{10\\baselineskip} ou
    \\vspace{8pt}, avec leur argument entre accolades (gestion des accolades
    imbriquées). Indispensable avant de chercher "le dernier nombre du
    texte" : sans ça, le "10" de \\needspace{10\\baselineskip} — une commande
    de mise en page qui précède presque chaque exercice dans les fichiers
    APMEP pour éviter une coupure de page — était pris à tort pour la
    réponse de l'exercice PRÉCÉDENT (repéré par un vrai cas : trois exercices
    de suite renvoyaient "10" au lieu de leur vraie réponse)."""
    for commande in COMMANDES_MISE_EN_PAGE:
        motif = re.compile(re.escape(commande) + r"\*?\s*\{")
        while True:
            m = motif.search(texte)
            if not m:
                break
            profondeur = 1
            i = m.end()
            while i < len(texte) and profondeur > 0:
                if texte[i] == "{":
                    profondeur += 1
                elif texte[i] == "}":
                    profondeur -= 1
                i += 1
            texte = texte[:m.start()] + texte[i:]
    return texte


def retirer_commentaires_latex(texte: str) -> str:
    """Retire tout ce qui suit un % non échappé sur chaque ligne (commentaire
    LaTeX), qu'il occupe la ligne entière ou seulement sa fin. Un vrai
    pourcentage s'écrit \\% (backslash devant) et n'est donc PAS retiré (la
    négation "non précédé d'un backslash" protège ce cas). Indispensable :
    un corrigé réel contenait "...\\np[~g]{59}. %arrondir à l'entier car 65..."
    — sans retirer ce commentaire de fin de ligne, le "65" qu'il contient
    aurait été pris à tort pour la réponse finale (59 était la vraie)."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ligne) for ligne in texte.splitlines())


def extraire_texte_tex(chemin: Path) -> str:
    """Lit un fichier .tex, retire tous les commentaires LaTeX (l'APMEP en
    truffe ses sources — par ex. un dessin TikZ laissé en commentaire à côté
    du PSTricks réellement utilisé — qui ne sont que du bruit pour la suite
    du traitement), puis retire les commandes de mise en page qui pourraient
    fausser la recherche de la réponse finale."""
    texte = chemin.read_text(encoding="utf-8")
    return retirer_commandes_mise_en_page(retirer_commentaires_latex(texte))


def extraire_texte_fichier(chemin: Path) -> str:
    if chemin.suffix.lower() == ".pdf":
        return extraire_texte_pdf(chemin)
    if chemin.suffix.lower() == ".tex":
        return extraire_texte_tex(chemin)
    raise ValueError(f"Extension non supportée : {chemin}")


# ---------------------------------------------------------------------------
# 2. Découpage en exercices
# ---------------------------------------------------------------------------

# Fonctionne à la fois sur du texte extrait d'un PDF et sur du LaTeX source :
# dans les deux cas, "Exercice" est presque toujours suivi directement du
# numéro (peu importe la commande qui l'entoure : \subsection*{Exercice 1...},
# \section*{Exercice 1 (3 points)}, \textbf{\textsc{Exercice 1} \hfill...}).
MOTIF_EXERCICE = re.compile(r"Exercice\s+(\d+)\b", re.IGNORECASE)


def decouper_en_exercices(texte: str) -> dict:
    """Renvoie {numéro_exercice (str) : texte_du_bloc}. Si un même numéro
    apparaît plusieurs fois, les occurrences sont concaténées plutôt
    qu'écrasées."""
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


def extraire_arguments_commande(texte: str, commande: str) -> list:
    """Extrait tous les arguments {...} d'une commande LaTeX donnée (ex.
    "\\fbox"), en gérant les accolades imbriquées : une regex naïve du genre
    r"\\fbox\\{(.*?)\\}" s'arrêterait à la première accolade fermante
    rencontrée, qui peut appartenir à une sous-commande interne (par exemple
    \\fbox{$A=\\dfrac{17}{12}$} contient déjà deux accolades internes)."""
    resultats = []
    for m in re.finditer(re.escape(commande) + r"\{", texte):
        debut = m.end()
        profondeur = 1
        i = debut
        while i < len(texte) and profondeur > 0:
            if texte[i] == "{":
                profondeur += 1
            elif texte[i] == "}":
                profondeur -= 1
            i += 1
        if profondeur == 0:
            resultats.append(texte[debut:i - 1])
    return resultats


def extraire_reponse_courte(texte_corrige_exercice: str) -> str:
    """Voir la documentation en tête de fichier pour l'ordre de préférence
    (\\fbox, puis dernier nombre du texte, puis dernière phrase). Renvoie une
    chaîne courte plutôt qu'une phrase entière avec ses calculs intermédiaires :
    sinon le contrôle qualité de generate_from_annales.py exigerait à tort que
    TOUS les nombres intermédiaires de cette phrase réapparaissent dans la
    conclusion du dialogue généré.

    Remarque sur le choix du "dernier nombre du texte" plutôt que du "dernier
    nombre juste après un signe =" : une conclusion du type "la masse d'une
    boule est d'environ \\np[~g]{59}" arrondit un résultat intermédiaire
    ($0,9\\times 65=58,5$) SANS "=" devant la valeur arrondie retenue — chercher
    spécifiquement après un "=" aurait donc à tort renvoyé 58,5 au lieu de 59,
    la vraie réponse officielle."""
    boites = extraire_arguments_commande(texte_corrige_exercice, r"\fbox")
    if boites:
        return boites[-1].strip()

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
    simple et déterministe. Renvoie "a_classifier" si rien ne correspond — à
    corriger à la main si besoin."""
    for motif, theme in MOTS_CLES_THEME:
        if motif.search(enonce):
            return theme
    return "a_classifier"


# ---------------------------------------------------------------------------
# 4. Association sujet / corrigé et construction du CSV
# ---------------------------------------------------------------------------

EXTENSIONS_SUPPORTEES = (".tex", ".pdf")


def trouver_paires(dossier: Path) -> dict:
    """Associe chaque <id>__sujet.(tex|pdf) à son <id>__corrige.(tex|pdf)
    (même dossier, convention de nom imposée — voir l'en-tête de ce fichier).
    Le sujet et le corrigé d'une même session n'ont pas besoin d'être dans le
    même format."""
    def indexer(suffixe_nom: str) -> dict:
        index = {}
        for ext in EXTENSIONS_SUPPORTEES:
            for p in dossier.glob(f"*{suffixe_nom}{ext}"):
                index[p.name[: -len(suffixe_nom + ext)]] = p
        return index

    sujets = indexer("__sujet")
    corriges = indexer("__corrige")

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
    texte_sujet = extraire_texte_fichier(chemin_sujet)
    texte_corrige = extraire_texte_fichier(chemin_corrige)

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
    parser.add_argument("--source", default="annales-source", help="Dossier contenant les fichiers <id>__sujet.(tex|pdf) / <id>__corrige.(tex|pdf)")
    parser.add_argument("--out", default="annales_apmep.csv", help="Fichier CSV de sortie")
    args = parser.parse_args()

    dossier = Path(args.source)
    if not dossier.is_dir():
        sys.exit(f"Dossier introuvable : {dossier}")

    paires = trouver_paires(dossier)
    if not paires:
        sys.exit(
            f"Aucune paire <id>__sujet.(tex|pdf) / <id>__corrige.(tex|pdf) trouvée dans {dossier}. "
            "Vérifie la convention de nommage (voir l'en-tête de ce script)."
        )

    toutes_les_lignes = []
    for session_id, (chemin_sujet, chemin_corrige) in sorted(paires.items()):
        print(f"[traitement] session {session_id} ({chemin_sujet.suffix} / {chemin_corrige.suffix})")
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
    print("Relis un échantillon de 'reponse_correcte' à la main avant de lancer generate_from_annales.py.")


if __name__ == "__main__":
    main()
