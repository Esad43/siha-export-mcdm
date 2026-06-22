"""
Sélection Stratégique des Marchés d'Exportation — SİHA Turcs
Modèle interactif Fuzzy AHP – TOPSIS
Ahmet Esad Haşimoğlu — Galatasaray Üniversitesi · 2026
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =====================================================================
# 1. DONNÉES DE BASE (Tableaux 4.2, 4.3, 4.4, 5.9 du mémoire)
# =====================================================================

PAYS = ["Pologne", "Roumanie", "Pays-Bas", "Tchéquie", "Hongrie", "Slovaquie"]
CODES = ["A1", "A2", "A3", "A4", "A5", "A6"]

SOUS_CRITERES = [
    ("K1.1", "Dépenses défense / PIB (%)",   "benefit", 22.57, "K1"),
    ("K1.2", "TCAC budget défense (%)",      "benefit", 19.28, "K1"),
    ("K1.3", "Part équipements OTAN (%)",    "benefit",  5.28, "K1"),
    ("K2.1", "Indice de Paix Mondial (GPI)", "cost",    11.86, "K2"),
    ("K2.2", "WGI Stabilité politique",      "benefit", 11.86, "K2"),
    ("K2.3", "Risque embargo / sanctions",   "cost",     4.21, "K2"),
    ("K3.1", "Indice CPI (corruption)",      "benefit",  5.65, "K3"),
    ("K3.2", "Obligation offset (%)",        "cost",     1.91, "K3"),
    ("K3.3", "Facilité licence SSB",         "benefit",  1.91, "K3"),
    ("K4.1", "Note Moody's (num.)",          "benefit",  4.21, "K4"),
    ("K4.2", "Inflation HICP 2023 (%)",      "cost",     1.46, "K4"),
    ("K5.1", "Nombre de concurrents",        "cost",     2.52, "K5"),
    ("K5.2", "Score sécurité ToT / PI",      "benefit",  7.26, "K5"),
]

CODES_CRIT = [c[0] for c in SOUS_CRITERES]
LIBELLES   = {c[0]: c[1] for c in SOUS_CRITERES}
TYPES      = {c[0]: c[2] for c in SOUS_CRITERES}
POIDS_BASE = {c[0]: c[3] for c in SOUS_CRITERES}
PARENT     = {c[0]: c[4] for c in SOUS_CRITERES}

CRIT_PRINCIPAUX = {
    "K1": ("Potentiel du marché", 47.14),
    "K2": ("Risque géopolitique", 27.94),
    "K3": ("Cadre institutionnel", 9.47),
    "K4": ("Environnement économique", 5.67),
    "K5": ("Environnement concurrentiel", 9.78),
}

GROUPES = {
    "K1": ["K1.1", "K1.2", "K1.3"],
    "K2": ["K2.1", "K2.2", "K2.3"],
    "K3": ["K3.1", "K3.2", "K3.3"],
    "K4": ["K4.1", "K4.2"],
    "K5": ["K5.1", "K5.2"],
}

MATRICE_BRUTE = {
    "K1.1": [4.12, 2.26, 2.05, 2.10, 2.14, 2.20],
    "K1.2": [26.4, 11.7, 14.4, 27.6, 19.0, 11.3],
    "K1.3": [36.2, 30.1, 28.5, 23.4, 28.6, 35.8],
    "K2.1": [1.678, 1.832, 1.527, 1.459, 1.502, 1.644],
    "K2.2": [67.5, 56.2, 63.9, 82.5, 64.9, 72.2],
    "K2.3": [2, 2, 4, 3, 2, 2],
    "K3.1": [54, 46, 79, 57, 42, 54],
    "K3.2": [100, 80, 0, 100, 30, 50],
    "K3.3": [5, 4, 3, 4, 4, 3],
    "K4.1": [15, 13, 20, 18, 12, 15],
    "K4.2": [11.5, 10.4, 4.1, 10.7, 17.6, 10.5],
    "K5.1": [4, 5, 6, 4, 3, 3],
    "K5.2": [7, 6, 9, 8, 5, 7],
}

INFOS_PAYS = {
    "Pologne":   ("A1", 38.00, 4.12, "Acheté", "24 TB2 livrés (2022-2024) ; 2e contrat signé oct. 2024"),
    "Roumanie":  ("A2",  8.00, 2.26, "Acheté", "18 TB2 ; livraisons achevées février 2024"),
    "Pays-Bas":  ("A3", 23.60, 2.05, "Intérêt", "Acheteur potentiel ; retrait Reaper prévu 2025"),
    "Tchéquie":  ("A4",  9.85, 2.10, "Intérêt", "Interet officiel documente ; non partenaire Eurodrone"),
    "Hongrie":   ("A5",  5.23, 2.14, "Négociation", "Relations bilaterales favorables ; acheteur potentiel"),
    "Slovaquie": ("A6",  3.07, 2.20, "Annulé", "Negociation 2022, annulee 2024 ; contact documente"),
}

# =====================================================================
# 2. FONCTION TOPSIS
# =====================================================================

def topsis(poids_dict):
    codes = CODES_CRIT
    X = np.array([MATRICE_BRUTE[c] for c in codes], dtype=float)
    w = np.array([poids_dict[c] for c in codes], dtype=float)
    w = w / w.sum()

    R = X / np.sqrt((X**2).sum(axis=1, keepdims=True))
    T = R * w[:, None]

    A_pos = np.zeros(len(codes))
    A_neg = np.zeros(len(codes))
    for i, c in enumerate(codes):
        if TYPES[c] == "benefit":
            A_pos[i], A_neg[i] = T[i].max(), T[i].min()
        else:
            A_pos[i], A_neg[i] = T[i].min(), T[i].max()

    S_pos = np.sqrt(((T - A_pos[:, None])**2).sum(axis=0))
    S_neg = np.sqrt(((T - A_neg[:, None])**2).sum(axis=0))
    C = S_neg / (S_pos + S_neg)

    df = pd.DataFrame({
        "Pays": PAYS, "Code": CODES,
        "S+ (dist. A*)": S_pos, "S- (dist. A-)": S_neg,
        "C* (proximité)": C,
    })
    df = df.sort_values("C* (proximité)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rang", range(1, len(df) + 1))
    return df


def scenario_poids(nom):
    p = dict(POIDS_BASE)
    if nom == "S1 — K1 réduit 30%":
        for c in GROUPES["K1"]: p[c] *= 0.70
    elif nom == "S2 — K1 augmenté 30%":
        for c in GROUPES["K1"]: p[c] *= 1.30
    elif nom == "S3 — Institutions/Éco dominantes":
        for c in GROUPES["K3"] + GROUPES["K4"]: p[c] *= 3.0
    elif nom == "S4 — Investisseur prudent":
        p["K3.1"] *= 3.0; p["K4.1"] *= 3.0
        for c in GROUPES["K1"]: p[c] *= 0.5
    elif nom == "S5 — Critères experts exclus":
        for c in ["K2.3", "K3.3", "K5.2"]: p[c] = 0.0
    elif nom == "S6 — Poids égaux":
        for c in CODES_CRIT: p[c] = 100.0 / len(CODES_CRIT)
    elif nom == "S7 — Risque géopolitique prioritaire":
        p["K2.1"] *= 3.0
    tot = sum(p.values())
    return {c: p[c] / tot * 100 for c in p}


# =====================================================================
# 3. CONFIG & EN-TÊTE
# =====================================================================

st.set_page_config(
    page_title="Sélection Marchés SİHA — Fuzzy AHP/TOPSIS",
    page_icon="🛰️", layout="wide",
)

st.title("🛰️ Sélection Stratégique des Marchés d'Exportation pour les SİHA Turcs")
st.caption(
    "Modèle interactif Fuzzy AHP – TOPSIS · Ahmet Esad Haşimoğlu · "
    "Galatasaray Üniversitesi · 2026"
)

tab_intro, tab_crit, tab_pays, tab_calc, tab_sens = st.tabs([
    "📖 Présentation",
    "📋 Critères",
    "🌍 Pays candidats",
    "⚙️ Classement interactif",
    "🔬 Analyse de sensibilité",
])

# ---------------------------------------------------------------------
# ONGLET 1 — PRÉSENTATION
# ---------------------------------------------------------------------
with tab_intro:
    st.header("Présentation du projet")
    st.markdown(
        """
L'industrie de défense turque connaît une croissance rapide : les exportations
d'armement sont passées de **1,6 à 6,73 milliards de dollars** en dix ans, et des
entreprises comme **Baykar** vendent leurs drones armés (**SİHA**) dans plus de
34 pays. Dans ce contexte, le **choix du marché d'exportation** devient une
décision stratégique majeure.

Ce projet propose un **Système d'Aide à la Décision (SAD)** combinant deux
méthodes de décision multicritère pour classer six marchés européens candidats :
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1️⃣ Fuzzy AHP (Buckley, 1985)")
        st.markdown(
            "Calcule les **poids des critères** à partir de jugements d'experts "
            "exprimés par des nombres flous triangulaires (NFT), pour intégrer "
            "l'incertitude des comparaisons."
        )
    with c2:
        st.subheader("2️⃣ TOPSIS (Hwang & Yoon, 1981)")
        st.markdown(
            "**Classe les pays** : la meilleure alternative est celle qui est la "
            "plus proche de la solution idéale positive et la plus éloignée de la "
            "solution idéale négative."
        )

    st.divider()
    st.subheader("📌 Comment utiliser cette application")
    st.markdown(
        """
- **Onglet « Critères »** : découvrez les 5 critères principaux et 13 sous-critères.
- **Onglet « Pays candidats »** : profil des six marchés européens analysés.
- **Onglet « Classement interactif »** : modifiez les poids des critères et observez
  le classement TOPSIS se recalculer en temps réel.
- **Onglet « Analyse de sensibilité »** : comparez les 7 scénarios de pondération du mémoire.
        """
    )
    st.info(
        "**Résultat de référence (mémoire, Ch. 5)** : Pologne 1re (C* = 0,8107), "
        "Tchéquie 2e (0,4895), Hongrie 3e, Pays-Bas 4e, Slovaquie 5e, Roumanie 6e."
    )

# ---------------------------------------------------------------------
# ONGLET 2 — CRITÈRES
# ---------------------------------------------------------------------
with tab_crit:
    st.header("Hiérarchie des critères d'évaluation")
    st.markdown(
        "Le modèle s'appuie sur **5 critères principaux** décomposés en "
        "**13 sous-critères**. Chaque sous-critère est de type **bénéfice** "
        "(à maximiser) ou **coût** (à minimiser)."
    )

    df_princ = pd.DataFrame({
        "Critère": [f"{k} — {v[0]}" for k, v in CRIT_PRINCIPAUX.items()],
        "Poids (%)": [v[1] for v in CRIT_PRINCIPAUX.values()],
    })
    fig_pie = px.pie(df_princ, names="Critère", values="Poids (%)",
                     title="Poids des 5 critères principaux (Fuzzy AHP)")
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Détail des sous-critères")
    for k, (nom, poids) in CRIT_PRINCIPAUX.items():
        with st.expander(f"**{k} — {nom}**  ·  poids global {poids:.2f} %", expanded=(k == "K1")):
            rows = []
            for code in GROUPES[k]:
                rows.append({
                    "Code": code,
                    "Sous-critère": LIBELLES[code],
                    "Type": "🟢 Bénéfice" if TYPES[code] == "benefit" else "🔴 Coût",
                    "Poids global (%)": f"{POIDS_BASE[code]:.2f}",
                })
            st.table(pd.DataFrame(rows))

    st.caption(
        "🟢 Bénéfice = plus la valeur est élevée, mieux c'est. "
        "🔴 Coût = plus la valeur est faible, mieux c'est."
    )

# ---------------------------------------------------------------------
# ONGLET 3 — PAYS CANDIDATS
# ---------------------------------------------------------------------
with tab_pays:
    st.header("Les six marchés candidats")
    st.markdown(
        "Ces pays ont été retenus après un **protocole d'élimination à six filtres** "
        "appliqué aux membres européens de l'OTAN (budget, engagement OTAN, "
        "stabilité, relations bilatérales, autonomie SİHA, ouverture du marché)."
    )

    df_pays = pd.DataFrame([
        {
            "Code": v[0], "Pays": pays,
            "Budget (Mrd USD)": v[1], "Déf./PIB (%)": v[2],
            "Statut TB2": v[3], "Justification": v[4],
        }
        for pays, v in INFOS_PAYS.items()
    ])
    st.dataframe(df_pays, use_container_width=True, hide_index=True)

    st.subheader("Budgets de défense comparés")
    fig_b = px.bar(df_pays.sort_values("Budget (Mrd USD)"),
                   x="Budget (Mrd USD)", y="Pays", orientation="h",
                   color="Budget (Mrd USD)", color_continuous_scale="Blues",
                   text="Budget (Mrd USD)")
    fig_b.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_b, use_container_width=True)

    st.caption("Sources : SIPRI Military Expenditure Database (2025) ; CAAT (2025) ; Baykar.")

# ---------------------------------------------------------------------
# ONGLET 4 — CLASSEMENT INTERACTIF
# ---------------------------------------------------------------------
with tab_calc:
    st.header("⚙️ Classement TOPSIS interactif")
    st.markdown(
        "Ajustez les **poids des critères** dans le panneau de gauche. "
        "Le classement se recalcule automatiquement."
    )

    if "poids" not in st.session_state:
        st.session_state.poids = dict(POIDS_BASE)

    st.sidebar.header("⚖️ Poids des critères (%)")

    # --- Bouton de réinitialisation aux ağırlıklar du mémoire ---
    if st.sidebar.button("🔄 Revenir aux ağırlıklar du mémoire", type="primary", use_container_width=True):
        st.session_state.poids = dict(POIDS_BASE)
        # Effacer les clés de slider pour forcer leur rechargement
        for code in CODES_CRIT:
            if f"sl_{code}" in st.session_state:
                del st.session_state[f"sl_{code}"]
        st.rerun()

    st.sidebar.divider()

    preset = st.sidebar.selectbox(
        "Ou appliquer un scénario prédéfini",
        ["Base (Ch. 5)", "S1 — K1 réduit 30%", "S2 — K1 augmenté 30%",
         "S6 — Poids égaux"],
    )
    if st.sidebar.button("Appliquer le scénario", use_container_width=True):
        if preset == "Base (Ch. 5)":
            st.session_state.poids = dict(POIDS_BASE)
        else:
            st.session_state.poids = scenario_poids(preset)
        for code in CODES_CRIT:
            if f"sl_{code}" in st.session_state:
                del st.session_state[f"sl_{code}"]
        st.rerun()

    st.sidebar.divider()

    nouveaux_poids = {}
    for code in CODES_CRIT:
        nouveaux_poids[code] = st.sidebar.slider(
            f"{code} — {LIBELLES[code]}",
            0.0, 40.0,
            float(round(st.session_state.poids[code], 2)),
            0.1, key=f"sl_{code}",
        )
    st.session_state.poids = nouveaux_poids
    total = sum(nouveaux_poids.values())
    st.sidebar.metric("Somme des poids", f"{total:.1f} %")
    st.sidebar.caption("Renormalisés à 100 % dans le calcul.")

    resultat = topsis(nouveaux_poids)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🏆 Classement")
        st.dataframe(
            resultat,
            use_container_width=True, hide_index=True,
            column_config={
                "S+ (dist. A*)": st.column_config.NumberColumn(format="%.5f"),
                "S- (dist. A-)": st.column_config.NumberColumn(format="%.5f"),
                "C* (proximité)": st.column_config.ProgressColumn(
                    format="%.4f", min_value=0.0, max_value=1.0),
            },
        )
        gagnant = resultat.iloc[0]
        st.success(f"**Marché prioritaire : {gagnant['Pays']}**  ·  C* = {gagnant['C* (proximité)']:.4f}")

    with col2:
        st.subheader("📊 Scores C*")
        fig = px.bar(resultat, x="C* (proximité)", y="Pays", orientation="h",
                     color="C* (proximité)", color_continuous_scale="Greens",
                     text="C* (proximité)")
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚖️ Répartition actuelle des poids")
    poids_norm = {c: nouveaux_poids[c] / total * 100 for c in CODES_CRIT}
    df_poids = pd.DataFrame({
        "Critère": [f"{c} — {LIBELLES[c]}" for c in CODES_CRIT],
        "Poids (%)": [poids_norm[c] for c in CODES_CRIT],
    })
    fig_p = px.bar(df_poids, x="Poids (%)", y="Critère", orientation="h",
                   color="Poids (%)", color_continuous_scale="Blues")
    fig_p.update_layout(height=450, showlegend=False,
                        yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_p, use_container_width=True)

# ---------------------------------------------------------------------
# ONGLET 5 — ANALYSE DE SENSIBILITÉ
# ---------------------------------------------------------------------
with tab_sens:
    st.header("🔬 Analyse de sensibilité")
    st.markdown(
        "TOPSIS est relancé sur les **sept scénarios de pondération** du chapitre 6 "
        "afin de tester la robustesse du classement."
    )

    SCENARIOS = [
        "Base (Ch. 5)", "S1 — K1 réduit 30%", "S2 — K1 augmenté 30%",
        "S3 — Institutions/Éco dominantes", "S4 — Investisseur prudent",
        "S5 — Critères experts exclus", "S6 — Poids égaux",
        "S7 — Risque géopolitique prioritaire",
    ]

    lignes, rangs = [], {p: [] for p in PAYS}
    for sc in SCENARIOS:
        p = POIDS_BASE if sc == "Base (Ch. 5)" else scenario_poids(sc)
        res = topsis(p)
        scores = {r["Pays"]: r["C* (proximité)"] for _, r in res.iterrows()}
        for _, r in res.iterrows():
            rangs[r["Pays"]].append(r["Rang"])
        ligne = {"Scénario": sc}
        ligne.update({pays: scores[pays] for pays in PAYS})
        ligne["1er rang"] = res.iloc[0]["Pays"]
        lignes.append(ligne)
    df_sens = pd.DataFrame(lignes)

    st.subheader("Scores C* par scénario")
    st.dataframe(
        df_sens, use_container_width=True, hide_index=True,
        column_config={p: st.column_config.NumberColumn(format="%.4f") for p in PAYS},
    )

    st.subheader("Évolution des scores à travers les scénarios")
    fig_line = go.Figure()
    for pays in PAYS:
        fig_line.add_trace(go.Scatter(
            x=df_sens["Scénario"], y=df_sens[pays],
            mode="lines+markers", name=pays))
    fig_line.update_layout(yaxis_title="Score C*", xaxis_title="Scénario",
                           height=500, legend_title="Pays", xaxis_tickangle=-30)
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Indice de stabilité du rang")
    rang_base = {r["Pays"]: r["Rang"] for _, r in topsis(POIDS_BASE).iterrows()}
    stab = []
    for pays in PAYS:
        rs = rangs[pays]
        occ = sum(1 for r in rs if r == rang_base[pays])
        stab.append({
            "Pays": pays, "Rang de base": rang_base[pays],
            "Occurrences même rang": f"{occ} / {len(rs)}",
            "Indice de stabilité": f"{occ/len(rs)*100:.1f} %",
            "Rang min-max": f"{min(rs)}e - {max(rs)}e",
        })
    st.dataframe(pd.DataFrame(stab).sort_values("Rang de base"),
                 use_container_width=True, hide_index=True)

    st.info(
        "La **Roumanie** reste 6e dans tous les scénarios (faiblesse structurelle). "
        "Le sommet est disputé entre **Pologne**, **Pays-Bas** et **Tchéquie** selon "
        "que l'on privilégie le potentiel budgétaire ou la solvabilité institutionnelle."
    )

st.divider()
st.caption(
    "⚠️ Modèle académique · données 2023-2024 (SIPRI, OTAN, IEP, Banque Mondiale, "
    "Transparency International, Moody's, Eurostat) · Projet de fin d'études."
)
