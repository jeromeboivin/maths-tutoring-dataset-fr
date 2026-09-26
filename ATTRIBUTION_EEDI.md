# Attribution requise (dataset Eedi)

Les dialogues marqués `"source": "eedi"` dans ce jeu de données sont traduits et adaptés à partir de Eedi/Question-Anchored-Tutoring-Dialogues-2k, publié sous licence Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
https://creativecommons.org/licenses/by-nc/4.0/

**Usage NON COMMERCIAL uniquement** — comme SocraTeach, plus restrictif que MathDial (CC BY-SA). Si tu envisages un jour de distribuer commercialement le modèle fine-tuné ou l'app qui l'utilise, il faudra exclure ces dialogues du jeu d'entraînement (filtre facile : `"source" != "eedi"`).

Dépôt d'origine (Hugging Face) :
https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k

Rappel important : contrairement à MathDial/SocraTeach, la bonne réponse n'est pas garantie par un contrôle automatique fiable pour les QCM à options textuelles (voir l'en-tête de translate_eedi.py) — relis à la main un échantillon plus large que d'habitude avant d'utiliser ces dialogues pour l'entraînement.
