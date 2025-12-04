import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Estimateur Immobilier", layout="wide")

# --- Chargement et prétraitement (caché pour réactivité) ---
@st.cache_data
def load_and_prepare(path: str):
    df = pd.read_csv(path)

    # Mapping yes/no -> 1/0
    yes_no_map = {"yes": 1, "no": 0}
    cols_yes_no = [
        "mainroad", "guestroom", "basement", "hotwaterheating",
        "airconditioning", "prefarea"
    ]
    df[cols_yes_no] = df[cols_yes_no].replace(yes_no_map)

    # Mapping furnishingstatus
    furn_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    df["furnishingstatus"] = df["furnishingstatus"].replace(furn_map)

    # Features order (même que le notebook)
    feature_cols = [
        "area", "bedrooms", "bathrooms", "stories",
        "mainroad", "guestroom", "basement", "hotwaterheating",
        "airconditioning", "parking", "prefarea", "furnishingstatus"
    ]

    X = df[feature_cols].values
    y = df[["price"]].values  # 2D pour scaler y

    return df, feature_cols, X, y

df, FEATURE_COLS, X_all, y_all = load_and_prepare("./Housing.csv")

# --- Entraînement du modèle Lasso sur l'ensemble des données (au lancement) ---
@st.cache_resource
def train_model(X, y, alpha=0.1):
    # Standardisation des features et de la target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).ravel()  # Lasso attend 1D target

    # Entraînement Lasso (alpha paramétrable)
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_scaled, y_scaled)

    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = train_model(X_all, y_all, alpha=0.1)

# --- Sidebar : Estimateur instantané ---
st.sidebar.header("Estimateur instantané — Entrées utilisateur")

# fonctions utilitaires pour valeurs par défaut
def num_default(col):
    return float(df[col].median())

def int_default(col):
    return int(df[col].median())

# Numeric inputs
area = st.sidebar.number_input(
    "Surface (area)",
    min_value=int(df["area"].min()),
    max_value=int(df["area"].max()),
    value=int_default("area"),
    step=100
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    int(df["bedrooms"].min()),
    int(df["bedrooms"].max()),
    int_default("bedrooms")
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    int(df["bathrooms"].min()),
    int(df["bathrooms"].max()),
    int_default("bathrooms")
)

stories = st.sidebar.slider(
    "Stories",
    int(df["stories"].min()),
    int(df["stories"].max()),
    int_default("stories")
)

parking = st.sidebar.slider(
    "Parking",
    int(df["parking"].min()),
    int(df["parking"].max()),
    int_default("parking")
)

# Binary checkboxes (defaults based on median)
def checkbox_default(col):
    return bool(df[col].median() >= 0.5)

mainroad = st.sidebar.checkbox("Main road", value=checkbox_default("mainroad"))
guestroom = st.sidebar.checkbox("Guest room", value=checkbox_default("guestroom"))
basement = st.sidebar.checkbox("Basement", value=checkbox_default("basement"))
hotwaterheating = st.sidebar.checkbox("Hot water heating", value=checkbox_default("hotwaterheating"))
airconditioning = st.sidebar.checkbox("Air conditioning", value=checkbox_default("airconditioning"))
prefarea = st.sidebar.checkbox("Preferred area", value=checkbox_default("prefarea"))

# Furnishing status selectbox (user-friendly labels)
furn_labels = ["unfurnished", "semi-furnished", "furnished"]
# default based on median encoded value
furn_default_encoded = int(df["furnishingstatus"].median())
furn_default_label = furn_labels[furn_default_encoded]
furnishingstatus_label = st.sidebar.selectbox(
    "Furnishing status",
    options=furn_labels,
    index=furn_labels.index(furn_default_label)
)

# mapping label -> encoded
furn_map_label_to_code = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
furnishingstatus = furn_map_label_to_code[furnishingstatus_label]

# Bouton prédire (optionnel)
if st.sidebar.button("Estimer le prix"):
    pass  # trigger rerun; prediction en continu ci-dessous

# --- Préparation des features pour prédiction ---
x_user = np.array([[
    area, bedrooms, bathrooms, stories,
    1 if mainroad else 0,
    1 if guestroom else 0,
    1 if basement else 0,
    1 if hotwaterheating else 0,
    1 if airconditioning else 0,
    parking,
    1 if prefarea else 0,
    furnishingstatus
]])

# Normaliser et prédire
x_user_scaled = scaler_X.transform(x_user)
y_user_scaled_pred = model.predict(x_user_scaled)  # 1D
y_user_pred = scaler_y.inverse_transform(y_user_scaled_pred.reshape(-1, 1)).ravel()[0]

# --- Zone principale : résultats et visualisations ---
st.title("Estimateur Immobilier — Analyse instantanée et position sur le marché")

# Affichage du prix estimé
st.metric(label="Prix estimé", value=f"${y_user_pred:,.0f}")

# Jauge (plotly) : position du prix estimé par rapport à min/moy/max
price_min = float(df["price"].min())
price_mean = float(df["price"].mean())
price_max = float(df["price"].max())

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=y_user_pred,
    number={'prefix': '$', 'valueformat': ',.0f'},
    title={'text': "Position du prix estimé vs marché"},
    gauge={
        'axis': {'range': [price_min, price_max]},
        'steps': [
            {'range': [price_min, price_mean], 'color': "lightgreen"},
            {'range': [price_mean, price_max], 'color': "lightcoral"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': price_mean
        }
    }
))
gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

# Scatter plot Area vs Price + point utilisateur
scatter = px.scatter(
    df,
    x="area",
    y="price",
    labels={"area": "Area", "price": "Price"},
    title="Area vs Price — Position de la maison simulée",
    opacity=0.6,
    height=500
)

# Ajouter point utilisateur (rouge, bien visible)
scatter.add_trace(go.Scatter(
    x=[area],
    y=[y_user_pred],
    mode="markers",
    marker=dict(color="red", size=14, symbol="x"),
    name="Maison simulée"
))

scatter.update_layout(margin=dict(l=20, r=20, t=50, b=20))

# Disposition : jauge à gauche, scatter à droite
col1, col2 = st.columns([1, 2])
with col1:
    st.plotly_chart(gauge, use_container_width=True)
with col2:
    st.plotly_chart(scatter, use_container_width=True)

# Afficher résumé des features entrées
with st.expander("Voir les caractéristiques saisies"):
    inputs_df = pd.DataFrame(x_user, columns=FEATURE_COLS)
    # reconvertir furnishingstatus en label pour lecture
    inv_furn = {v: k for k, v in furn_map_label_to_code.items()}
    inputs_df["furnishingstatus"] = inputs_df["furnishingstatus"].map(inv_furn)
    st.write(inputs_df.T)

# Petit résumé de performance / info du modèle (optionnel)
with st.expander("Infos modèle"):
    st.write("Modèle : Lasso (alpha=0.1) entraîné sur l'ensemble des données.")
    # montrer coefficients (non transformés) — interprétation limitée à cause du scaling
    coefs = model.coef_
    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coef (scaled space)": coefs
    })
    st.dataframe(coef_df)

# Footer minimal
st.caption("Application créée pour estimer le prix d'une maison et visualiser sa position sur le marché. Données chargées depuis Housing.csv.")