# Dataset de tutorat en mathématiques (français)

Objectif : construire un jeu de données de conversations élève / professeur
virtuel, en français, pour fine-tuner un petit modèle open-weight (type
Qwen 1,5B-3B) capable de tourner sur iPad et d'accompagner un élève en
mathématiques — du primaire au lycée — **sans jamais donner directement la
réponse**, tout en restant bienveillant face à l'insistance de l'élève.

Voir [`plan_dataset_maths_fr.md`](plan_dataset_maths_fr.md) pour le cadrage
complet (principes pédagogiques, format des données, carte des thèmes par
niveau, méthode de génération, volume visé).

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `plan_dataset_maths_fr.md` | Document de cadrage du projet |
| `seed_problems.csv` | Banque de problèmes de départ (niveau, thème, énoncé, réponse, erreur typique) |
| `exemples.jsonl` | 3 dialogues écrits à la main, comme référence de style |
| `generate_dataset.py` | Génère des dialogues **à partir de zéro** via l'API OpenAI (`gpt-6-luna`), avec un contrôle qualité automatique et un renforcement explicite de la résistance à la demande de réponse directe + bienveillance |
| `translate_mathdial.py` | Traduit un échantillon du dataset public [MathDial](https://github.com/eth-nlped/mathdial) (CC BY-SA 4.0) |
| `translate_socrateach.py` | Traduit un échantillon du dataset public [SocraTeach / SocraticLM](https://github.com/Ljyustc/SocraticLM) (CC BY-NC 4.0) |
| `translate_eedi.py` | Traduit un échantillon du dataset public [Eedi](https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k) (CC BY-NC 4.0), avec adaptation au format QCM |
| `eedi-source/` | Petit échantillon réel (6 conversations) pour tester `translate_eedi.py` sans avoir à télécharger tout le dataset |
| `ATTRIBUTION.md`, `ATTRIBUTION_SOCRATEACH.md`, `ATTRIBUTION_EEDI.md` | Attributions et conditions de licence à conserver pour chaque source |

## Licences des sources publiques utilisées

| Source | Licence | Usage commercial ? |
|---|---|---|
| MathDial | CC BY-SA 4.0 | Oui, avec partage à l'identique |
| SocraTeach (SocraticLM) | CC BY-NC 4.0 | **Non** |
| Eedi | CC BY-NC 4.0 | **Non** |

➜ Si le modèle fine-tuné ou l'app qui l'utilise devait un jour être distribué
commercialement, il faudra exclure les dialogues issus de SocraTeach et
Eedi (filtre simple sur le champ `"source"` de chaque ligne JSONL).

## Récupérer les datasets sources complets

Ces gros dépôts externes ne sont **pas** versionnés ici (données tierces,
volumineuses). Pour les récupérer avant de lancer les scripts de traduction
à grande échelle :

```bash
git clone --depth 1 https://github.com/eth-nlped/mathdial.git mathdial-source
git clone --depth 1 https://github.com/Ljyustc/SocraticLM.git socraticlm-source
```

Pour Eedi (hébergé sur Hugging Face), télécharge directement les CSV :

```bash
curl -LO https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k/resolve/main/anchored-dialogues.csv
curl -LO https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k/resolve/main/dq-question-metadata.csv
```

## Utilisation

```bash
pip install openai
export OPENAI_API_KEY="votre-clé"

# Génération de dialogues originaux
python3 generate_dataset.py --seed seed_problems.csv --out dialogues_generes.jsonl --variations 3

# Traduction des datasets publics
python3 translate_mathdial.py --source mathdial-source/data/train.jsonl --out mathdial_fr.jsonl --n 50
python3 translate_socrateach.py --source socraticlm-source/data/SocraTeach_multi.json --out socrateach_fr.jsonl --n 50
python3 translate_eedi.py --dialogues eedi-source/anchored-dialogues.csv --metadata eedi-source/dq-question-metadata.csv --out eedi_fr.jsonl
```

## Prochaines étapes

Voir la section « Prochaines étapes concrètes » de
[`plan_dataset_maths_fr.md`](plan_dataset_maths_fr.md).
