import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="House Predict Pro", layout="wide")

# --- 1. CHARGEMENT DES ARTEFACTS & DATA ---
@st.cache_resource
def load_data():
    try:
        artefacts = joblib.load('modele_housing.pkl')
        df_raw = pd.read_csv('Housing.csv')
        return artefacts, df_raw
    except FileNotFoundError as e:
        return None, None

artefacts, df_raw = load_data()

if artefacts is None or df_raw is None:
    st.error("Erreur critique : Fichiers manquants ('modele_housing.pkl' ou 'Housing.csv').")
    st.stop()

# Extraction des paramètres du modèle
theta = artefacts.get("modele_poids", artefacts.get("theta"))
mean_x = artefacts.get("x_mean")
std_x = artefacts.get("x_std")
scaler_y = artefacts.get("y_scaler")


# --- 2. FONCTIONS MÉTIER ---

def predict_price_manual(features_brutes):
    """Calcule le prix selon les poids de la descente de gradient."""
    x_input = np.array(features_brutes)
    x_normalized = (x_input - mean_x) / std_x
    x_with_bias = np.insert(x_normalized, 0, 1.0)
    prediction_norm = x_with_bias @ theta
    
    # Reshape pour scaler inverse
    if isinstance(prediction_norm, np.ndarray):
        val_to_reshape = prediction_norm.reshape(-1, 1)
    else:
        val_to_reshape = np.array([[prediction_norm]])
        
    price_dollar = scaler_y.inverse_transform(val_to_reshape)
    return max(0, price_dollar[0][0]) # Pas de prix négatif

def get_similar_houses(user_features, df, top_n=5):
    """Trouve les maisons les plus proches dans le dataset (Nearest Neighbors basique)."""
    # 1. Convertir le DF en numérique pour le calcul de distance
    df_num = df.copy()
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df_num[col] = df_num[col].apply(lambda x: 1 if x == 'yes' else 0)
    
    furn_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    df_num['furnishingstatus'] = df_num['furnishingstatus'].map(furn_map)
    
    # Colonnes features dans l'ordre exact du modèle (sans le prix)
    feature_cols = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", 
                    "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]
    
    # 2. Calculer la distance Euclidienne pondérée
    X_data = df_num[feature_cols].values
    X_user = np.array(user_features)
    
    # Distance euclidienne standardisée
    dist = np.linalg.norm((X_data - mean_x) / std_x - (X_user - mean_x) / std_x, axis=1)
    
    df_res = df.iloc[np.argsort(dist)[:top_n]].copy()
    df_res['similarity_score'] = dist[np.argsort(dist)[:top_n]]
    return df_res


# --- 3. INTERFACE UTILISATEUR (SIDEBAR) ---
st.sidebar.title("House Predict Pro")
st.sidebar.markdown("---")

def get_user_input():
    st.sidebar.header("Caractéristiques")
    
    # Layout compact
    c1, c2 = st.sidebar.columns(2)
    area = c1.number_input("Surface (sqft)", 1000, 16000, 4000, 100)
    stories = c2.selectbox("Étages", [1, 2, 3, 4], index=1)
    
    c3, c4 = st.sidebar.columns(2)
    bedrooms = c3.selectbox("Chambres", [1, 2, 3, 4, 5, 6], index=2)
    bathrooms = c4.selectbox("Salles de Bain", [1, 2, 3, 4], index=0)
    
    parking = st.sidebar.slider("Parking", 0, 3, 1)

    st.sidebar.header("Standing")
    furnishing_options = {"Meublé": 2, "Semi-meublé": 1, "Non meublé": 0}
    furn_label = st.sidebar.radio("Ameublement", list(furnishing_options.keys()), horizontal=True)
    furnishing_val = furnishing_options[furn_label]

    st.sidebar.header("Options")
    # Utilisation d'expander pour ne pas encombrer
    with st.sidebar.expander("Détails techniques", expanded=True):
        mainroad = 1 if st.checkbox("Route Principale", True) else 0
        prefarea = 1 if st.checkbox("Quartier Prisé", False) else 0
        aircon = 1 if st.checkbox("Climatisation", True) else 0
        guestroom = 1 if st.checkbox("Chambre d'amis", False) else 0
        basement = 1 if st.checkbox("Sous-sol", False) else 0
        hotwater = 1 if st.checkbox("Chauffe-eau", False) else 0

    features = [area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                hotwater, aircon, parking, prefarea, furnishing_val]
    
    return features, {
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "stories": stories,
        "mainroad": mainroad, "guestroom": guestroom, "basement": basement,
        "hotwater": hotwater, "aircon": aircon, "parking": parking,
        "prefarea": prefarea, "furnishing_val": furnishing_val
    }

user_features_list, user_inputs_dict = get_user_input()

# --- 4. DASHBOARD CENTRAL ---

# Calculs
predicted_price = predict_price_manual(user_features_list)
price_per_sqft = predicted_price / user_inputs_dict['area']

# --- HEADER : RÉSULTAT CLÉ EN MAIN ---
col_main, col_gauge = st.columns([2, 1])

with col_main:
    st.subheader("Estimation de Valeur")
    st.markdown(f"<h1 style='font-size: 60px; color: #4CAF50;'>${predicted_price:,.0f}</h1>", unsafe_allow_html=True)
    st.caption(f"Soit environ **${price_per_sqft:.2f}/sqft** pour ce bien de {user_inputs_dict['area']} sqft.")
    
    # Export des données (CSV)
    export_df = pd.DataFrame([user_inputs_dict])
    export_df['Estimated_Price'] = predicted_price
    csv = export_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Télécharger le rapport CSV",
        data=csv,
        file_name='estimation_immobiliere.csv',
        mime='text/csv',
    )

with col_gauge:
    # Jauge simplifiée et efficace
    min_p, max_p = df_raw['price'].min(), df_raw['price'].max()
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = predicted_price,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [min_p, max_p]}, 'bar': {'color': "#4CAF50"}}
    ))
    fig_gauge.update_layout(height=150, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# --- ONGLETS FONCTIONNELS ---
tab1, tab2, tab3, tab4 = st.tabs(["Investissement", "Plus-Value", "Comparables", "Analyse Avancée"])

# === ONGLET 1 : INVESTISSEMENT LOCATIF (REMPLACE FINANCEMENT) ===
with tab1:
    st.subheader("Analyse de Rentabilité Locative")
    st.markdown("Estimation des revenus potentiels pour un investisseur.")
    
    col_inv1, col_inv2 = st.columns(2)
    with col_inv1:
        target_yield = st.slider("Rendement Brut Cible (%)", 2.0, 12.0, 6.0, 0.5)
        occupancy = st.slider("Taux d'occupation annuel (%)", 50, 100, 95)
        maintenance_cost = st.number_input("Entretien annuel estimé ($)", value=int(predicted_price * 0.01), step=100)
    
    with col_inv2:
        # Calculs de rendement
        annual_gross_rent = predicted_price * (target_yield / 100)
        monthly_rent_target = annual_gross_rent / 12
        
        # Revenu net estimé (Loyer brut - Vacance - Entretien)
        effective_annual_income = (annual_gross_rent * (occupancy / 100)) - maintenance_cost
        
        st.metric("Loyer Mensuel Conseillé", f"${monthly_rent_target:,.0f}", delta="Cible")
        st.metric("Cashflow Net Annuel Estimé", f"${effective_annual_income:,.0f}", 
                 delta=f"{effective_annual_income/predicted_price*100:.1f}% Net Yield")
        
        st.caption(f"*Basé sur une vacance de {100-occupancy}% et des coûts d'entretien de ${maintenance_cost}.")
        if effective_annual_income > 0:
            st.success("Ce bien présente un potentiel de cashflow positif.")
        else:
            st.warning("Attention : Les coûts risquent de dépasser les revenus locatifs.")

# === ONGLET 2 : SIMULATEUR ROI ===
with tab2:
    st.subheader("Potentiel de Rénovation")
    
    scenarios = []
    
    # 1. Ajout Clim
    if user_inputs_dict['aircon'] == 0:
        feats_ac = user_features_list.copy()
        feats_ac[8] = 1 
        price_ac = predict_price_manual(feats_ac)
        scenarios.append({"Action": "Installer la Climatisation", "Gain": price_ac - predicted_price})
        
    # 2. Rénovation Meubles
    if user_inputs_dict['furnishing_val'] < 2:
        feats_fur = user_features_list.copy()
        feats_fur[11] = 2 
        price_fur = predict_price_manual(feats_fur)
        scenarios.append({"Action": "Vendre entièrement meublé", "Gain": price_fur - predicted_price})
        
    # 3. Ajout SDB
    feats_bath = user_features_list.copy()
    feats_bath[2] += 1
    price_bath = predict_price_manual(feats_bath)
    scenarios.append({"Action": "Ajouter 1 Salle de Bain", "Gain": price_bath - predicted_price})

    # 4. Ajout Chambre
    feats_bed = user_features_list.copy()
    feats_bed[1] += 1
    price_bed = predict_price_manual(feats_bed)
    scenarios.append({"Action": "Créer une chambre supplémentaire", "Gain": price_bed - predicted_price})

    if not scenarios:
        st.info("Ce bien est déjà très optimisé !")
    else:
        max_gain = max([s['Gain'] for s in scenarios]) if scenarios else 1.0
        if max_gain == 0: max_gain = 1.0
        
        cols_sim = st.columns(len(scenarios))
        for i, scen in enumerate(scenarios):
            with cols_sim[i]:
                st.metric(label=scen["Action"], value=f"+ ${scen['Gain']:,.0f}", delta="Plus-value")
                st.progress(min(1.0, scen['Gain'] / (max_gain * 1.2)))

# === ONGLET 3 : COMPARABLES ===
with tab3:
    st.subheader("Biens Similaires Vendus")
    similar_df = get_similar_houses(user_features_list, df_raw)
    st.dataframe(
        similar_df[['price', 'area', 'bedrooms', 'bathrooms', 'airconditioning', 'furnishingstatus']],
        column_config={
            "price": st.column_config.NumberColumn("Prix", format="$%d"),
            "area": st.column_config.NumberColumn("Surface", format="%d sqft"),
            "airconditioning": "Clim",
            "furnishingstatus": "État"
        },
        use_container_width=True,
        hide_index=True
    )
    
    avg_sim_price = similar_df['price'].mean()
    st.info(f"Prix moyen des comparables : ${avg_sim_price:,.0f}")

# === ONGLET 4 : ANALYSE AVANCÉE (NOUVEAU) ===
with tab4:
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.subheader("Sensibilité Prix / Surface")
        st.caption("Comment évolue le prix de VOTRE configuration si la surface change ?")
        
        # Génération courbe de sensibilité
        areas_to_test = np.linspace(1500, 12000, 50)
        prices_sensitivity = []
        
        base_features = list(user_features_list)
        for a in areas_to_test:
            temp_feats = base_features.copy()
            temp_feats[0] = a # Index 0 est area
            prices_sensitivity.append(predict_price_manual(temp_feats))
            
        fig_sens = px.line(x=areas_to_test, y=prices_sensitivity, labels={'x': 'Surface (sqft)', 'y': 'Prix Estimé ($)'})
        # Marquer la position actuelle
        fig_sens.add_trace(go.Scatter(x=[user_inputs_dict['area']], y=[predicted_price], mode='markers', marker=dict(color='red', size=10), name="Actuel"))
        st.plotly_chart(fig_sens, use_container_width=True)

    with col_a2:
        st.subheader("Impact des Variables (Poids)")
        weights = theta[1:].flatten()
        features_labels = ["Surface", "Chambres", "SDB", "Étages", "Route", "Amis", "Sous-sol", "Eau Ch.", "Clim", "Parking", "Quartier", "Meubles"]
        
        fig_bar = px.bar(x=weights, y=features_labels, orientation='h', title="Importance relative dans le modèle")
        st.plotly_chart(fig_bar, use_container_width=True)