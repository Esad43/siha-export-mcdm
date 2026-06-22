"""
Sélection Stratégique des Marchés d'Exportation — SİHA Turcs
Modèle interactif Fuzzy AHP – TOPSIS
Ahmet Esad Haşimoğlu — Galatasaray Üniversitesi
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =====================================================================
# 1. DONNÉES DE BASE (issues du mémoire — Tableaux 4.3, 4.4, 5.9)
# =====================================================================

# Alternatives (pays)
PAYS = ["Pologne", "Roumanie", "Pays-Bas", "Tchéquie", "Hongrie", "Slovaquie"]
CODES = ["A1", "A2", "A3", "A4", "A5", "A6"]

# Sous-critères : code, libellé, type ("benefit" ou "cost"), poids global de base (%)
SOUS_CRITERES = [
    ("K1.1", "Dépenses défense / PIB (%)",        "benefit", 22.57),
    ("K1.2", "TCAC budget défense (%)",           "benefit", 19.28),
    ("K1.3", "Part équipements OTAN (%)",         "benefit",  5.28),
    ("K2.1", "Indice de Paix Mondial (GPI)",      "cost",    11.86),
    ("K2.2", "WGI Stabilité politique",           "benefit", 11.86),
    ("K2.3", "Risque embargo / sanctions",        "cost",     4.21),
    ("K3.1", "Indice CPI",                        "benefit",  5.65),
    ("K3.2", "Obligation offset (%)",             "cost",     1.91),
    ("K3.3", "Facilité licence SSB",              "benefit",  1.91),
    ("K4.1", "Note Moody's (num.)",               "benefit",  4.21),
    ("K4.2", "Inflation HICP 2023 (%)",           "cost",     1.46),
    ("K5.1", "Nombre de concurrents",             "cost",     2.52),
    ("K5.2", "Score sécurité ToT / PI",           "benefit",  7.26),
]

CODES_CRIT = [c[0] for c in SOUS_CRITERES]
LIBELLES   = {c[0]: c[1] for c in SOUS_CRITERES}
TYPES      = {c[0]: c[2] for c in SOUS_CRITERES}
POIDS_BASE = {c[0]: c[3] for c in SOUS_CRITERES}

# Matrice de décision brute (Tableau 4.4) — lignes = critères, colonnes = pays
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

# Regroupement des sous-critères par critère principal (pour les presets)
GROUPES = {
    "K1": ["K1.1", "K1.2", "K1.3"],
    "K2": ["K2.1", "K2.2", "K2.3"],
    "K3": ["K3.1", "K3.2", "K3.3"],
    "K4": ["K4.1", "K4.2"],
    "K5": ["K5.1", "K5.2"],
}

# =====================================================================
# 2. FONCTION TOPSIS
# =====================================================================

def topsis(poids_dict):
    """
    Applique TOPSIS avec les poids fournis (dict code -> poids en %).
    Retourne un DataFrame trié avec S+, S-, C* et le rang.
    """
    codes = CODES_CRIT
    # Matrice (critères x pays)
    X = np.array([MATRICE_BRUTE[c] for c in codes], dtype=float)

    # Poids normalisés (somme = 1)
    w = np.array([poids_dict[c] for c in codes], dtype=float)
    w = w / w.sum()

    # Étape 1 — Normalisation vectorielle (par ligne = par critère)
    norm = np.sqrt((X**2).sum(axis=1, keepdims=True))
    R = X / norm

    # Étape 2 — Matrice pondérée
    T = R * w[:, None]

    # Étape 3 — Solutions idéales
    A_pos = np.zeros(len(codes))
    A_neg = np.zeros(len(codes))
    for i, c in enumerate(codes):
        if TYPES[c] == "benefit":
            A_pos[i] = T[i].max()
            A_neg[i] = T[i].min()
        else:  # cost
            A_pos[i] = T[i].min()
            A_neg[i] = T[i].max()

    # Étape 4 — Distances euclidiennes
    S_pos = np.sqrt(((T - A_pos[:, None])**2).sum(axis=0))
    S_neg = np.sqrt(((T - A_neg[:, None])**2).sum(axis=0))

    # Coefficient de proximité
    C = S_neg / (S_pos + S_neg)

    df = pd.DataFrame({
        "Pays": PAYS,
        "Code": CODES,
        "S+ (dist. A*)": S_pos,
        "S- (dist. A⁻)": S_neg,
        "C* (proximité)": C,
    })
    df = df.sort_values("C* (proximité)", ascending=False).reset_index(drop=True)
    df.insert(0, "Rang", range(1, len(df) + 1))
    return df


# =====================================================================
# 3. INTERFACE STREAMLIT
# =====================================================================

st.set_page_config(
    page_title="Sélection Marchés SİHA — Fuzzy AHP/TOPSIS",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Sélection Stratégique des Marchés d'Exportation pour les SİHA Turcs")
st.caption(
    "Modèle interactif Fuzzy AHP – TOPSIS · Ahmet Esad Haşimoğlu · "
    "Galatasaray Üniversitesi · 2026"
)

st.markdown(
    "Ajustez les **poids des critères** dans le panneau de gauche. "
    "Le classement TOPSIS des six pays se recalcule automatiquement."
)

# ---- Initialisation des poids dans la session ----
if "poids" not in st.session_state:
    st.session_state.poids = dict(POIDS_BASE)

# ---- SIDEBAR : réglage des poids ----
st.sidebar.header("⚖️ Poids des critères (%)")

# Presets (scénarios du chapitre 6)
preset = st.sidebar.selectbox(
    "Scénario prédéfini",
    [
        "Personnalisé / Base (Ch. 5)",
        "S1 — K1 réduit de 30 %",
        "S2 — K1 augmenté de 30 %",
        "S6 — Poids égaux (1/13)",
        "Réinitialiser (Base)",
    ],
)

def appliquer_preset(nom):
    p = dict(POIDS_BASE)
    if nom == "S1 — K1 réduit de 30 %":
        for c in GROUPES["K1"]:
            p[c] *= 0.70
    elif nom == "S2 — K1 augmenté de 30 %":
        for c in GROUPES["K1"]:
            p[c] *= 1.30
    elif nom == "S6 — Poids égaux (1/13)":
        for c in CODES_CRIT:
            p[c] = 100.0 / len(CODES_CRIT)
    # renormalisation à 100 %
    tot = sum(p.values())
    for c in p:
        p[c] = p[c] / tot * 100
    return p

if st.sidebar.button("Appliquer le scénario"):
    if preset.startswith("Réinitialiser") or preset.startswith("Personnalisé"):
        st.session_state.poids = dict(POIDS_BASE)
    else:
        st.session_state.poids = appliquer_preset(preset)

st.sidebar.divider()

# Sliders par critère
nouveaux_poids = {}
for code in CODES_CRIT:
    nouveaux_poids[code] = st.sidebar.slider(
        f"{code} — {LIBELLES[code]}",
        min_value=0.0,
        max_value=40.0,
        value=float(round(st.session_state.poids[code], 2)),
        step=0.1,
        key=f"sl_{code}",
    )

st.session_state.poids = nouveaux_poids
total_poids = sum(nouveaux_poids.values())
st.sidebar.metric("Somme des poids (avant normalisation)", f"{total_poids:.1f} %")
st.sidebar.caption(
    "Les poids sont automatiquement renormalisés à 100 % dans le calcul TOPSIS."
)

# ---- CALCUL TOPSIS ----
resultat = topsis(nouveaux_poids)

# ---- AFFICHAGE PRINCIPAL ----
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🏆 Classement TOPSIS")
    st.dataframe(
        resultat.style.format({
            "S+ (dist. A*)": "{:.5f}",
            "S- (dist. A⁻)": "{:.5f}",
            "C* (proximité)": "{:.4f}",
        }).background_gradient(subset=["C* (proximité)"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
    )
    vainqueur = resultat.iloc[0]["Pays"]
    score = resultat.iloc[0]["C* (proximité)"]
    st.success(f"**Marché prioritaire : {vainqueur}** (C* = {score:.4f})")

with col2:
    st.subheader("📊 Scores de proximité C*")
    fig = px.bar(
        resultat,
        x="C* (proximité)",
        y="Pays",
        orientation="h",
        color="C* (proximité)",
        color_continuous_scale="Greens",
        text="C* (proximité)",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- POIDS ACTUELS (graphique) ----
st.subheader("⚖️ Répartition actuelle des poids")
poids_norm = {c: nouveaux_poids[c] / total_poids * 100 for c in CODES_CRIT}
df_poids = pd.DataFrame({
    "Critère": [f"{c} — {LIBELLES[c]}" for c in CODES_CRIT],
    "Poids (%)": [poids_norm[c] for c in CODES_CRIT],
})
fig_p = px.bar(df_poids, x="Poids (%)", y="Critère", orientation="h",
               color="Poids (%)", color_continuous_scale="Blues")
fig_p.update_layout(height=450, showlegend=False,
                    yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_p, use_container_width=True)

# =====================================================================
# 4. ANALYSE DE SENSIBILITÉ (chapitre 6)
# =====================================================================

st.divider()
st.header("🔬 Analyse de sensibilité")
st.markdown(
    "Cette section relance TOPSIS sur les **sept scénarios de pondération** "
    "du chapitre 6 et compare les classements."
)

def scenario_poids(nom):
    p = dict(POIDS_BASE)
    if nom == "S1 — K1 réduit 30%":
        for c in GROUPES["K1"]: p[c] *= 0.70
    elif nom == "S2 — K1 augmenté 30%":
        for c in GROUPES["K1"]: p[c] *= 1.30
    elif nom == "S3 — Institutions/Éco dominantes":
        # K3 et K4 portés à 50 % du total
        for c in GROUPES["K3"] + GROUPES["K4"]: p[c] *= 3.0
    elif nom == "S4 — Investisseur prudent":
        # priorité CPI + Moody's
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

SCENARIOS = [
    "Base (Ch. 5)",
    "S1 — K1 réduit 30%",
    "S2 — K1 augmenté 30%",
    "S3 — Institutions/Éco dominantes",
    "S4 — Investisseur prudent",
    "S5 — Critères experts exclus",
    "S6 — Poids égaux",
    "S7 — Risque géopolitique prioritaire",
]

# Tableau des scores C* par scénario
lignes = []
for sc in SCENARIOS:
    p = POIDS_BASE if sc == "Base (Ch. 5)" else scenario_poids(sc)
    res = topsis(p)
    scores = {row["Pays"]: row["C* (proximité)"] for _, row in res.iterrows()}
    premier = res.iloc[0]["Pays"]
    ligne = {"Scénario": sc}
    ligne.update({pays: scores[pays] for pays in PAYS})
    ligne["1er rang"] = premier
    lignes.append(ligne)

df_sens = pd.DataFrame(lignes)

st.subheader("Scores C* par scénario")
st.dataframe(
    df_sens.style.format({pays: "{:.4f}" for pays in PAYS})
                 .background_gradient(subset=PAYS, cmap="RdYlGn"),
    use_container_width=True,
    hide_index=True,
)

# Graphique : évolution du rang par pays
st.subheader("Évolution des scores C* à travers les scénarios")
fig_line = go.Figure()
for pays in PAYS:
    fig_line.add_trace(go.Scatter(
        x=df_sens["Scénario"],
        y=df_sens[pays],
        mode="lines+markers",
        name=pays,
    ))
fig_line.update_layout(
    yaxis_title="Score C*",
    xaxis_title="Scénario",
    height=500,
    legend_title="Pays",
    xaxis_tickangle=-30,
)
st.plotly_chart(fig_line, use_container_width=True)

# Indice de stabilité (proportion de scénarios au rang de base)
st.subheader("Indice de stabilité du rang")
rangs = {pays: [] for pays in PAYS}
for sc in SCENARIOS:
    p = POIDS_BASE if sc == "Base (Ch. 5)" else scenario_poids(sc)
    res = topsis(p)
    for _, row in res.iterrows():
        rangs[row["Pays"]].append(row["Rang"])

rang_base = {row["Pays"]: row["Rang"] for _, row in topsis(POIDS_BASE).iterrows()}
stab = []
for pays in PAYS:
    rs = rangs[pays]
    occ = sum(1 for r in rs if r == rang_base[pays])
    stab.append({
        "Pays": pays,
        "Rang de base": rang_base[pays],
        "Occurrences même rang": f"{occ} / {len(rs)}",
        "Indice de stabilité": f"{occ/len(rs)*100:.1f} %",
        "Rang min–max": f"{min(rs)}e – {max(rs)}e",
    })
df_stab = pd.DataFrame(stab).sort_values("Rang de base")
st.dataframe(df_stab, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "⚠️ Modèle académique basé sur des données 2023–2024 (SIPRI, OTAN, IEP, "
    "Banque Mondiale, Transparency International, Moody's, Eurostat). "
    "Projet de fin d'études — usage pédagogique."
)
