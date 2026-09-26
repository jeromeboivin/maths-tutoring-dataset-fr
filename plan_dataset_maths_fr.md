# Construire un dataset de tutorat en mathématiques (français)

Objectif : obtenir un jeu de données de conversations élève / professeur virtuel, en français, pour entraîner un petit modèle (type Qwen 1,5B-3B) à accompagner un élève en maths — du primaire au lycée — sans jamais donner directement la réponse.

## 1. Le principe pédagogique à respecter

C'est le point qui fait la différence entre un "bon" dataset et un simple question-réponse. Deux travaux de recherche (MathDial, SocraticLM) convergent sur les mêmes règles :

- **Ne jamais donner la réponse à la première question.** Le professeur virtuel pose des questions, donne des indices, avant d'expliquer.
- **Réagir à l'erreur précise de l'élève**, pas à une erreur générique. Si l'élève confond périmètre et aire, le prof doit pointer *cette* confusion, pas relancer dans le vague.
- **Des indices progressifs** : d'abord une relance légère ("qu'est-ce que tu cherches exactement ?"), puis un indice plus ciblé, puis — seulement si l'élève reste bloqué après plusieurs essais — une explication complète.
- **Types de relances du professeur** (repris de MathDial) :
  - *Recentrer* : ramener l'élève sur le bon point du problème
  - *Sonder* : poser une question ouverte pour comprendre où l'élève en est
  - *Indice général* : donner une piste sans viser l'erreur exacte
  - *Expliquer* : donner l'information directement (dernier recours)
- **Des profils d'élèves variés** : bonne méthode mais erreur de calcul, mauvaise méthode, confusion de vocabulaire, blocage total, élève qui devine sans réfléchir...

## 2. Format du fichier d'entraînement

Un fichier `.jsonl` (une conversation complète par ligne) :

```json
{"id": "col-042", "niveau": "college", "theme": "proportionnalite", "messages": [
  {"role": "system", "content": "Tu es un professeur de maths pour un élève de collège. Tu ne donnes jamais la réponse directement : tu poses des questions et donnes des indices progressifs."},
  {"role": "user", "content": "Pour une recette de 4 personnes il faut 250g de farine. Pour 6 personnes il faut combien ?"},
  {"role": "assistant", "content": "Bonne question ! Avant de calculer, dis-moi : est-ce que la quantité de farine augmente proportionnellement au nombre de personnes, ou pas forcément ?"},
  {"role": "user", "content": "Oui je pense que oui"},
  {"role": "assistant", "content": "Exactement. Donc si tu passes de 4 à 6 personnes, par quel nombre as-tu multiplié le nombre de personnes ?"}
]}
```

- `system` fixe le cadre une fois (niveau + règle du jeu)
- alternance `user` (élève) / `assistant` (professeur)
- une conversation peut durer de 2 à 10+ échanges

Ce format est accepté tel quel par Unsloth, LLaMA-Factory et la plupart des outils de fine-tuning.

## 3. Carte des thèmes par niveau (programmes officiels, réforme 2026)

**Primaire (cycle 2-3, CP à CM2)**
Numération et les quatre opérations, calcul mental, fractions simples (à partir du CM1), grandeurs et mesures (longueurs, masses, durées, monnaie), résolution de problèmes, repérage et figures planes.

**Collège (cycle 4, 5e à 3e)**
Nombres relatifs, calcul littéral (la lettre comme inconnue, introduite progressivement), fractions et proportionnalité, puissances et nombres premiers, théorèmes de Pythagore et Thalès, vecteurs (introduits en 3e), statistiques et probabilités.

**Lycée (2nde à Terminale)**
2nde : automatismes de calcul, fonctions (généralités), géométrie repérée. 1ère (spécialité) : dérivation, suites, second degré (discriminant), probabilités. Terminale : limites, intégrales, probabilités conditionnelles (ou maths complémentaires/expertes selon la filière).

*(La banque de problèmes de départ, `seed_problems.csv`, ne couvre pour l'instant qu'un échantillon de ces thèmes — à étoffer avant une génération à grande échelle.)*

## 4. Méthode de génération retenue

Une IA généraliste (Claude, GPT...) est capable de jouer les deux rôles à la fois si on le lui demande clairement, à partir d'un problème + d'une erreur-cible :

1. Piocher une ligne dans `seed_problems.csv` (niveau, thème, énoncé, réponse correcte, erreur typique)
2. Demander à l'IA de rédiger un dialogue complet où l'élève commet *cette* erreur précise, et où le professeur le guide sans donner la solution
3. **Contrôle qualité automatique** sur chaque dialogue généré :
   - la réponse correcte apparaît-elle bien à la fin, obtenue *par l'élève* ?
   - le professeur n'a-t-il pas donné la solution dès le 1er ou 2e message ?
   - le vocabulaire correspond-il au niveau visé ?
4. Écarter ou régénérer les dialogues qui ratent ces critères
5. **Relecture humaine** sur un échantillon (10-15 % recommandé) avant de lancer l'entraînement — surtout pour vérifier que les maths sont justes

## 5. Volume visé

Pour un modèle de la taille de Qwen 1,5B/3B, quelques milliers d'exemples propres et variés (2 000 à 5 000) suffisent déjà à changer nettement le comportement du modèle — mieux vaut ce volume avec de la diversité (niveaux × thèmes × profils d'erreurs) que 20 000 dialogues répétitifs.

## 6. Prochaines étapes concrètes

1. Étoffer `seed_problems.csv` (aujourd'hui : quelques exemples par niveau, à faire grossir — idéalement avec l'aide d'un professeur ou de manuels scolaires)
2. Lancer une génération pilote de 50 à 100 dialogues avec `generate_dataset.py`, les relire à la main, ajuster le prompt si besoin
3. Lancer la génération à volume complet
4. Vérifier le fichier final (encodage, doublons, longueur des conversations) avant de le donner à l'outil de fine-tuning
