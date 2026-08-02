# AdoptAI - Supervision Intelligente

Projet de stage 2026 visant à collecter les métriques système et prédire les ralentissements des postes Windows grâce au Machine Learning.

## Structure du Projet
- `src/`: Contient l'agent de collecte de données.
- `data/`: Stockage local de la base de données SQLite.
- `notebooks/`: Espace d'exploration de données (EDA) et modélisation.

## Comment Lancer l'Agent
1. Activez l'environnement : `source adoptai_env/bin/activate`
2. Lancez le script : `python src/collector.py`