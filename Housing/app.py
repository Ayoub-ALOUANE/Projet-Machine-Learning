import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Chargement des artefacts du modèle (déjà présent)
artefacts = joblib.load('./modele_housing.pkl')
theta = artefacts.get("modele_poids", artefacts.get("theta", None))
mean_x = artefacts.get("x_mean", None)
std_x = artefacts.get("x_std", None)
scaler_y = artefacts.get("y_scaler", None)

# Chemin du dataset (même dossier ou chemin relatif)
DATA_PATH = Path(__file__).parent / "Housing.csv"

# --- Fonctions utilitaires --------------------------------------------------
def load_dataset(path):
    """Charge le dataset Housing.csv et normalise les noms de colonnes."""
    df = pd.read_csv(path)
    # uniformiser noms colonnes
    df.columns = [c.strip() for c in df.columns]
    return df

def detect_price_column(df):
    """Trouve la colonne prix dans le dataset."""
    for candidate in ["price", "Price", "PRICE", "SalePrice"]:
        if candidate in df.columns:
            return candidate
    # fallback: la colonne numérique avec la plus grande variance
    num = df.select_dtypes(include=[np.number])
    if "price" in num.columns:
        return "price"
    if not num.empty:
        return num.var().idxmax()
    raise RuntimeError("Impossible de détecter la colonne prix dans le dataset.")

def prepare_input_df(df):
    """Prépare une table d'exemples de caractéristiques (applique get_dummies sur catégorielles).
       Retourne df_preproc et la liste ordonnée de colonnes d'entrée utilisées pour le modèle."""
    X = df.copy()
    price_col = detect_price_column(df)
    if price_col in X.columns:
        X = X.drop(columns=[price_col])
    # Colonnes booléennes encodées 'yes'/'no'
    for col in X.select_dtypes(include=['object']).columns:
        vals = X[col].dropna().unique()
        if set(map(str.lower, vals)) <= {"yes", "no", "y", "n"}:
            X[col] = X[col].astype(str).str.lower().map(lambda v: 1 if v in ("yes","y") else 0)
    # One-hot pour categoric non-binaires
    X = pd.get_dummies(X, drop_first=False)
    return X

def build_feature_vector(user_inputs, template_columns):
    """Construit un vecteur de caractéristiques aligné sur template_columns.
       user_inputs: dict col->value (après encodage si besoin)
    """
    x = pd.DataFrame([user_inputs])
    x = pd.get_dummies(x, drop_first=False)
    # aligner colonnes
    for col in template_columns:
        if col not in x.columns:
            x[col] = 0
    # Supprimer colonnes inattendues
    extra = [c for c in x.columns if c not in template_columns]
    if extra:
        x = x.drop(columns=extra)
    x = x[template_columns]
    return x.values.flatten()

def apply_scaling_and_predict(x_raw):
    """Applique centrage/normalisation avec mean_x/std_x puis prédit et remet à l'échelle y.
       Retourne un scalaire float propre.
    """
    if mean_x is None or std_x is None or theta is None:
        st.error("Les artefacts du modèle sont incomplets. Vérifiez le fichier 'modele_housing.pkl'.")
        return None

    # convertir en numpy 1D
    x_raw = np.asarray(x_raw, dtype=float).ravel()

    # gérer theta/intercept
    theta_arr = np.asarray(theta, dtype=float).ravel()
    if theta_arr.size == x_raw.size + 1:
        intercept = float(theta_arr[0])
        weights = theta_arr[1:]
    elif theta_arr.size == x_raw.size:
        intercept = 0.0
        weights = theta_arr
    else:
        # tenter d'adapter en tronquant ou en complétant avec zéros
        if theta_arr.size > x_raw.size:
            intercept = float(theta_arr[0])
            weights = theta_arr[1:1 + x_raw.size]
        else:
            intercept = 0.0
            weights = np.pad(theta_arr, (0, x_raw.size - theta_arr.size), 'constant')

    # standardiser avec mean_x/std_x (aligner tailles si nécessaire)
    mean_x_arr = np.asarray(mean_x, dtype=float).ravel()
    std_x_arr = np.asarray(std_x, dtype=float).ravel()

    if mean_x_arr.size > x_raw.size:
        mean_x_arr = mean_x_arr[:x_raw.size]
        std_x_arr = std_x_arr[:x_raw.size]
    elif mean_x_arr.size < x_raw.size:
        mean_x_arr = np.pad(mean_x_arr, (0, x_raw.size - mean_x_arr.size), 'constant')
        std_x_arr = np.pad(std_x_arr, (0, x_raw.size - std_x_arr.size), 'constant') + 1.0

    std_x_arr[std_x_arr == 0] = 1.0
    x_scaled = (x_raw - mean_x_arr) / std_x_arr

    y_scaled = float(intercept + np.dot(x_scaled, weights))

    # remettre y à l'échelle originale si scaler_y fourni
    y_orig = y_scaled
    try:
        if hasattr(scaler_y, "inverse_transform"):
            # sklearn scalers attend (n_samples, n_features)
            inv = scaler_y.inverse_transform(np.atleast_2d(y_scaled).T)
            # inv peut être (1,1) ou (1,) -> extraire premier élément
            y_orig = np.asarray(inv).ravel()[0]
        elif isinstance(scaler_y, dict):
            mean_y = scaler_y.get("mean_", scaler_y.get("y_mean", 0.0))
            scale_y = scaler_y.get("scale_", scaler_y.get("y_scale", 1.0))
            y_orig = y_scaled * scale_y + mean_y
        else:
            y_orig = float(y_scaled)
    except Exception:
        # si inverse_transform échoue, fallback sur valeur non transformée
        y_orig = float(y_scaled)

    # garantir un scalaire Python float
    y_arr = np.asarray(y_orig)
    if y_arr.size == 0:
        return None
    return float(y_arr.ravel()[0])

# --- Interface utilisateur --------------------------------------------------
st.set_page_config(page_title="Estimateur Immobilier", layout="wide")
st.title("Estimateur Immobilier — Instantané & Analyse du Marché")

# Charger dataset
try:
    df = load_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Impossible de charger '{DATA_PATH.name}': {e}")
    st.stop()

price_col = detect_price_column(df)
prices = df[price_col].dropna().astype(float)

# Préparer colonnes d'entrée modèle à partir du dataset
X_template = prepare_input_df(df)
template_cols = list(X_template.columns)

# Sidebar: inputs utilisateur (valeurs par défaut calculées depuis le dataset)
with st.sidebar:
    st.header("Estimateur Instantané")
    st.caption("Saisissez les caractéristiques de la maison")

    # Détecter colonnes numériques utilisables pour l'UI (exclure colonnes de dummies)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if price_col in numeric_cols:
        numeric_cols.remove(price_col)

    # Prioritaires pour l'affichage: common features
    prefer = ["area", "Area", "sqft", "total_sqft", "bedrooms", "Beds", "bathrooms", "baths", "stories", "parking"]
    shown = []
    user_inputs = {}

    # Pour chaque préférence, si présente dans df, afficher un number_input
    for p in prefer:
        if p in df.columns and p not in shown:
            col = df[p].dropna()
            default = float(col.median()) if not col.empty else 0.0
            step = max(0.1, float(col.std())/10) if not col.empty else 1.0
            user_val = st.number_input(label=p, value=float(default), step=step, format="%.2f")
            user_inputs[p] = user_val
            shown.append(p)

    # Afficher autres numériques restants de façon compacte
    for col in numeric_cols:
        if col in shown:
            continue
        colseries = df[col].dropna()
        default = float(colseries.median()) if not colseries.empty else 0.0
        user_val = st.number_input(label=col, value=float(default), step=1.0)
        user_inputs[col] = user_val
        shown.append(col)

    # Colonnes catégorielles/object -> selectbox avec valeurs (ou checkbox si binaire string)
    for col in df.select_dtypes(include=['object']).columns:
        vals = df[col].dropna().unique()
        vals_clean = [str(v).strip() for v in vals]
        if set(map(str.lower, vals_clean)) <= {"yes", "no", "y", "n"}:
            # checkbox binaire
            default_checked = (pd.Series(vals_clean).mode().iloc[0].lower() in ("yes", "y"))
            user_bool = st.checkbox(col, value=default_checked)
            user_inputs[col] = 1 if user_bool else 0
        else:
            # selectbox
            default = pd.Series(vals_clean).mode().iloc[0] if len(vals_clean) > 0 else ""
            choice = st.selectbox(col, options=vals_clean, index=0 if default == vals_clean[0] else 0)
            user_inputs[col] = choice

    st.markdown("---")
    predict_button = st.button("Estimer le prix")

# Construire vecteur aligné avec colonnes template
x_vector = build_feature_vector(user_inputs, template_cols)

# Si clic sur Estimer ou afficher estimation immédiate
if predict_button:
    prix_estime = apply_scaling_and_predict(x_vector)
    if prix_estime is None:
        st.stop()

    # Affichage principal
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Prix estimé", value=f"{prix_estime:,.0f} {''}")
        st.write("Caractéristiques saisies :")
        st.json(user_inputs)

    # Visualisation 1 : Jauge indiquant position du prix estimé par rapport au marché
    min_p, mean_p, max_p = float(prices.min()), float(prices.mean()), float(prices.max())
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prix_estime,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Position sur le marché"},
        delta={'reference': mean_p, 'valueformat':".0f"},
        gauge={
            'axis': {'range': [min_p, max_p]},
            'steps': [
                {'range': [min_p, mean_p], 'color': "lightgray"},
                {'range': [mean_p, max_p], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prix_estime}
        }
    ))
    with col2:
        st.plotly_chart(gauge, use_container_width=True)

    # Visualisation 2 : Scatter Area vs Price avec point utilisateur
    area_candidates = [c for c in df.columns if c.lower() in ("area", "total_sqft", "sqft", "area_sqft")]
    if area_candidates:
        area_col = area_candidates[0]
        fig = px.scatter(df, x=area_col, y=price_col, labels={area_col: "Area", price_col: "Price"},
                         title="Area vs Price — Position du bien simulé")
        # extraire area valeur entrée si existante sinon utiliser moyenne
        area_val = user_inputs.get(area_col, df[area_col].median() if area_col in df.columns else df[area_col].median())
        fig.add_scatter(x=[area_val], y=[prix_estime], mode='markers', marker=dict(color='red', size=14, symbol='x'),
                        name='Mon bien')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Le dataset ne contient pas de colonne 'area' reconnue pour tracer le scatter Area vs Price.")

    # Statistiques rapides du marché
    st.subheader("Résumé du marché")
    st.write(f"Min: {min_p:,.0f} — Moyenne: {mean_p:,.0f} — Max: {max_p:,.0f}")
else:
    st.info("Entrez les caractéristiques dans la barre latérale et cliquez sur 'Estimer le prix' pour lancer la prédiction.")

