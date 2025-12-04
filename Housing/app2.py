import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Prédicteur Prix Immobilier",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CHARGEMENT DES DONNÉES ====================
@st.cache_data
def load_data():
    """Charge le dataset pour les statistiques et visualisations"""
    df = pd.read_csv('Housing.csv')
    
    # Encodage des variables catégorielles (comme dans le notebook)
    yes_no_map = {"yes": 1, "no": 0}
    cols_yes_no = ["mainroad", "guestroom", "basement", "hotwaterheating", 
                   "airconditioning", "prefarea"]
    df[cols_yes_no] = df[cols_yes_no].replace(yes_no_map)
    
    furn_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    df["furnishingstatus"] = df["furnishingstatus"].replace(furn_map)
    
    return df

@st.cache_resource
def load_model():
    """Charge le modèle entraîné et ses artefacts"""
    artefacts = joblib.load('modele_housing.pkl')
    return artefacts

# Chargement
try:
    df = load_data()
    artefacts = load_model()
    theta = artefacts["modele_poids"]
    mean_x = artefacts["x_mean"]
    std_x = artefacts["x_std"]
    scaler_y = artefacts["y_scaler"]
except Exception as e:
    st.error(f"⚠️ Erreur de chargement: {e}")
    st.stop()

# ==================== STATISTIQUES DU MARCHÉ ====================
prix_min = df['price'].min()
prix_max = df['price'].max()
prix_moyen = df['price'].mean()
prix_median = df['price'].median()

# ==================== SIDEBAR: ENTRÉES UTILISATEUR ====================
st.sidebar.title("🏡 Configuration de la Propriété")
st.sidebar.markdown("---")

# Section 1: Caractéristiques Principales
st.sidebar.subheader("📐 Dimensions & Structure")
area = st.sidebar.number_input(
    "Surface (m²)",
    min_value=1000,
    max_value=20000,
    value=int(df['area'].median()),
    step=100,
    help="Surface totale du terrain en m²"
)

bedrooms = st.sidebar.slider(
    "Nombre de chambres",
    min_value=1,
    max_value=6,
    value=int(df['bedrooms'].median()),
    help="Nombre de chambres à coucher"
)

bathrooms = st.sidebar.slider(
    "Nombre de salles de bain",
    min_value=1,
    max_value=4,
    value=int(df['bathrooms'].median())
)

stories = st.sidebar.slider(
    "Nombre d'étages",
    min_value=1,
    max_value=4,
    value=int(df['stories'].median())
)

parking = st.sidebar.slider(
    "Places de parking",
    min_value=0,
    max_value=3,
    value=int(df['parking'].median())
)

# Section 2: Équipements & Localisation
st.sidebar.markdown("---")
st.sidebar.subheader("✨ Équipements & Emplacement")

mainroad = st.sidebar.checkbox(
    "Route principale",
    value=True,
    help="Située sur une route principale"
)

guestroom = st.sidebar.checkbox(
    "Chambre d'hôtes",
    value=False
)

basement = st.sidebar.checkbox(
    "Sous-sol",
    value=False
)

hotwaterheating = st.sidebar.checkbox(
    "Chauffage eau chaude",
    value=False
)

airconditioning = st.sidebar.checkbox(
    "Climatisation",
    value=True,
    help="Système de climatisation installé"
)

prefarea = st.sidebar.checkbox(
    "Zone préférentielle",
    value=True,
    help="Situé dans une zone privilégiée"
)

# Section 3: État de la propriété
st.sidebar.markdown("---")
st.sidebar.subheader("🛋️ Ameublement")
furnishing_options = {
    "Meublé": 2,
    "Semi-meublé": 1,
    "Non meublé": 0
}
furnishingstatus = st.sidebar.select_slider(
    "État d'ameublement",
    options=list(furnishing_options.keys()),
    value="Semi-meublé"
)
furnishingstatus_encoded = furnishing_options[furnishingstatus]

# ==================== PRÉDICTION ====================
# Construction du vecteur d'entrée (ordre identique au notebook)
X_user = np.array([[
    area, bedrooms, bathrooms, stories,
    int(mainroad), int(guestroom), int(basement), 
    int(hotwaterheating), int(airconditioning),
    parking, int(prefarea), furnishingstatus_encoded
]])

# Normalisation avec les paramètres d'entraînement
X_user_norm = (X_user - mean_x) / std_x

# Ajout de la colonne bias (terme constant)
X_user_norm = np.hstack((X_user_norm, np.ones((X_user_norm.shape[0], 1))))

# Prédiction normalisée
prediction_norm = X_user_norm.dot(theta)

# Dénormalisation pour obtenir le prix réel
prediction_price = scaler_y.inverse_transform(prediction_norm.reshape(-1, 1))[0][0]

# ==================== AFFICHAGE PRINCIPAL ====================
st.title("🏠 Estimateur de Prix Immobilier")
st.markdown("### Analyse Prédictive & Positionnement Marché")

# Ligne de métriques
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Prix Estimé",
        f"{prediction_price:,.0f} €",
        delta=f"{((prediction_price - prix_moyen) / prix_moyen * 100):+.1f}% vs. moyenne"
    )

with col2:
    st.metric("📊 Prix Moyen Marché", f"{prix_moyen:,.0f} €")

with col3:
    st.metric("📉 Prix Minimum", f"{prix_min:,.0f} €")

with col4:
    st.metric("📈 Prix Maximum", f"{prix_max:,.0f} €")

st.markdown("---")

# ==================== VISUALISATION 1: JAUGE ====================
st.subheader("🎯 Positionnement sur le Marché")

# Création de la jauge avec Plotly
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=prediction_price,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Prix Estimé (€)", 'font': {'size': 24}},
    delta={'reference': prix_moyen, 'increasing': {'color': "green"}, 
           'decreasing': {'color': "orange"}},
    number={'font': {'size': 40}, 'valueformat': ",.0f"},
    gauge={
        'axis': {'range': [prix_min, prix_max], 'tickformat': ",.0f"},
        'bar': {'color': "#1f77b4"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [prix_min, prix_moyen], 'color': '#d4edda'},
            {'range': [prix_moyen, prix_max], 'color': '#fff3cd'}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': prix_median
        }
    }
))

fig_gauge.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=50, b=20),
    font={'family': "Arial"}
)

st.plotly_chart(fig_gauge, use_container_width=True)

# ==================== VISUALISATION 2: SCATTER PLOT ====================
st.markdown("---")
st.subheader("📍 Votre Propriété sur le Marché")

# Création du scatter plot du dataset
fig_scatter = px.scatter(
    df,
    x='area',
    y='price',
    opacity=0.5,
    color='bedrooms',
    size='bathrooms',
    hover_data=['bedrooms', 'bathrooms', 'stories', 'parking'],
    color_continuous_scale='Viridis',
    labels={'area': 'Surface (m²)', 'price': 'Prix (€)', 'bedrooms': 'Chambres'},
    title="Distribution des Prix en Fonction de la Surface"
)

# Ajout du point représentant la propriété de l'utilisateur
fig_scatter.add_scatter(
    x=[area],
    y=[prediction_price],
    mode='markers',
    marker=dict(
        size=20,
        color='red',
        symbol='star',
        line=dict(width=2, color='darkred')
    ),
    name='Votre Estimation',
    showlegend=True
)

fig_scatter.update_layout(
    height=500,
    hovermode='closest',
    xaxis_title="Surface (m²)",
    yaxis_title="Prix (€)",
    font=dict(size=12)
)

st.plotly_chart(fig_scatter, use_container_width=True)

# ==================== ANALYSE DÉTAILLÉE ====================
st.markdown("---")
st.subheader("🔍 Analyse Détaillée de Votre Propriété")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 📋 Récapitulatif")
    st.markdown(f"""
    - **Surface**: {area:,} m²
    - **Chambres**: {bedrooms}
    - **Salles de bain**: {bathrooms}
    - **Étages**: {stories}
    - **Parking**: {parking} place(s)
    - **Ameublement**: {furnishingstatus}
    """)

with col_b:
    st.markdown("#### ✅ Équipements")
    equipements = {
        "Route principale": mainroad,
        "Chambre d'hôtes": guestroom,
        "Sous-sol": basement,
        "Chauffage": hotwaterheating,
        "Climatisation": airconditioning,
        "Zone préférentielle": prefarea
    }
    
    for equip, present in equipements.items():
        icon = "✅" if present else "❌"
        st.markdown(f"{icon} {equip}")

# ==================== STATISTIQUES COMPARATIVES ====================
st.markdown("---")
st.subheader("📊 Statistiques Comparatives")

col_stat1, col_stat2, col_stat3 = st.columns(3)

with col_stat1:
    percentile = (df['price'] < prediction_price).mean() * 100
    st.info(f"🎯 **Percentile**: Votre propriété est plus chère que **{percentile:.1f}%** des propriétés du marché")

with col_stat2:
    similar_area = df[(df['area'] >= area - 500) & (df['area'] <= area + 500)]
    if len(similar_area) > 0:
        avg_similar = similar_area['price'].mean()
        st.info(f"🏘️ **Prix moyen similaire**: {avg_similar:,.0f} € (surface ±500 m²)")
    else:
        st.warning("Aucune propriété similaire trouvée")

with col_stat3:
    similar_rooms = df[df['bedrooms'] == bedrooms]
    if len(similar_rooms) > 0:
        avg_rooms = similar_rooms['price'].mean()
        st.info(f"🛏️ **Prix moyen {bedrooms} chambres**: {avg_rooms:,.0f} €")
    else:
        st.warning("Aucune propriété avec ce nombre de chambres")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🤖 Modèle de Machine Learning - Régression Linéaire avec Régularisation Lasso</p>
    <p>📈 Dataset: Housing Prices | 🎯 Coefficient de Détermination: 0.633</p>
</div>
""", unsafe_allow_html=True)