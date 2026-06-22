# 🛰️ Sélection Stratégique des Marchés d'Exportation — SİHA Turcs

Application web interactive du modèle **Fuzzy AHP – TOPSIS** développé dans le cadre
du projet de fin d'études en Génie Industriel à l'Université de Galatasaray.

**Auteur :** Ahmet Esad Haşimoğlu
**Directeur :** Doç. Dr. Murat Levent Demircan · 2026

## Description

L'application permet de :
- Ajuster interactivement les **poids des 13 sous-critères** (sliders)
- Recalculer en temps réel le **classement TOPSIS** des 6 pays candidats
  (Pologne, Roumanie, Pays-Bas, Tchéquie, Hongrie, Slovaquie)
- Visualiser l'**analyse de sensibilité** sur les 7 scénarios du mémoire
- Consulter l'**indice de stabilité** du classement

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Méthodologie

- **Fuzzy AHP (Buckley, 1985)** : pondération des critères
- **TOPSIS (Hwang & Yoon, 1981)** : classement des alternatives

Données : SIPRI, OTAN, IEP (GPI), Banque Mondiale (WGI), Transparency
International (CPI), Moody's, Eurostat — 2023/2024.
