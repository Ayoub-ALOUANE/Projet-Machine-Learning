import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF  # Nécessite pip install fpdf

def run():
    # Page title
    st.title("Car Predict Pro")
    # --- 1. CHARGEMENT DES ARTEFACTS & DATA ---
    @st.cache_resource
    def load_data():
        try:
            artefacts = joblib.load('Cars/modele_cars.pkl')
            df_raw = pd.read_csv('Cars/CarPrice.csv')
            return artefacts, df_raw
        except FileNotFoundError as e:
            return None, None

    artefacts, df_raw = load_data()

    if artefacts is None or df_raw is None:
        st.error("Erreur critique : Fichiers manquants ('modele_cars.pkl' ou 'CarPrice.csv').")
        st.stop()

    # Extraction des paramètres du modèle
    theta = artefacts.get("modele_poids", artefacts.get("theta"))
    mean_x = artefacts.get("x_mean")
    std_x = artefacts.get("x_std")
    scaler_y = artefacts.get("y_scaler")
    poly_features = artefacts.get("poly_features")
    feature_columns = artefacts.get("feature_columns", [])
    categorical_cols = artefacts.get("categorical_cols", [])
    encoders = artefacts.get("encoders", {})


    # --- 2. FONCTIONS MÉTIER ---

    def predict_price_manual(features_brutes):
        """Calcule le prix selon les poids de la descente de gradient avec features polynomiales."""
        # Convertir les features en array
        x_input = np.array(features_brutes).reshape(1, -1)

        # Debug: Check if fueltype is gas (index 15 in features array)
        if len(features_brutes) > 15 and features_brutes[15] == 1:  # fueltype_encoded = 1 means gas
            print(f"Debug: Gas selected! Features: {features_brutes}")
            print(f"Debug: Features shape: {x_input.shape}")

        # Appliquer les features polynomiales
        x_poly = poly_features.transform(x_input)

        # Normaliser
        x_normalized = (x_poly - mean_x) / std_x

        # Ajouter le biais
        x_with_bias = np.hstack([np.ones((x_normalized.shape[0], 1)), x_normalized])

        # Prédiction
        prediction_norm = x_with_bias @ theta

        # Reshape pour scaler inverse
        if isinstance(prediction_norm, np.ndarray):
            val_to_reshape = prediction_norm.reshape(-1, 1)
        else:
            val_to_reshape = np.array([[prediction_norm]])

        price_usd = scaler_y.inverse_transform(val_to_reshape)
        raw_price = price_usd[0][0]

        # Debug: print prediction details for troubleshooting
        if raw_price <= 0:
            print(f"Debug: Raw prediction = {raw_price}, normalized prediction = {prediction_norm[0][0]}")
            print(f"Debug: Input features: {features_brutes}")
            print(f"Debug: Poly features contains NaN: {np.isnan(x_poly).any()}")
            print(f"Debug: Normalized features contains NaN: {np.isnan(x_normalized).any()}")

        return max(0, raw_price) # Pas de prix négatif

    def get_similar_cars(user_features, df, top_n=5):
        """Trouve les voitures les plus proches dans le dataset."""
        df_num = df.copy()
        
        # Create brand column from CarName (like in the notebook)
        df_num['brand'] = df_num['CarName'].apply(lambda x: x.split(' ')[0].lower())

        # Encoder les colonnes catégorielles avec les mêmes encoders
        for col in categorical_cols:
            if col in df_num.columns and col in encoders:
                df_num[col + '_encoded'] = encoders[col].transform(df_num[col])

        # Utiliser les colonnes features du modèle
        X_data = df_num[feature_columns].values
        X_user = np.array(user_features).reshape(1, -1)

        # Appliquer la transformation polynomiale
        X_data_poly = poly_features.transform(X_data)
        X_user_poly = poly_features.transform(X_user)

        # Calculer la distance dans l'espace normalisé
        dist = np.linalg.norm((X_data_poly - mean_x) / std_x - (X_user_poly - mean_x) / std_x, axis=1)

        df_res = df.iloc[np.argsort(dist)[:top_n]].copy()
        df_res['similarity_score'] = dist[np.argsort(dist)[:top_n]]
        return df_res

    def create_pdf_report(user_inputs, price):
        """Génère un rapport PDF simple."""
        pdf = FPDF()
        pdf.add_page()

        # Titre
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Rapport d'Estimation Automobile", ln=True, align='C')
        pdf.ln(10)

        # Prix
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Prix Estime: ${price:,.0f}", ln=True)
        pdf.ln(10)

        # Caractéristiques
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Details du vehicule:", ln=True)
        pdf.set_font("Arial", '', 11)

        for key, value in user_inputs.items():
            # Nettoyage basique pour compatibilité latin-1
            clean_key = str(key).replace('_', ' ').title()
            clean_val = str(value)
            pdf.cell(0, 8, f"{clean_key}: {clean_val}", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 10, "Genere par Car Price Predict Pro", ln=True, align='R')

        return pdf.output(dest='S').encode('latin-1')


    # --- 3. INTERFACE UTILISATEUR (SIDEBAR) ---
    st.sidebar.title("Car Price Predict Pro")
    st.sidebar.markdown("---")

    def get_user_input():
        st.sidebar.header("Caractéristiques")

        # Layout compact pour les caractéristiques principales
        c1, c2 = st.sidebar.columns(2)
        brand_options = df_raw['CarName'].apply(lambda x: x.split(' ')[0].lower()).unique() if df_raw is not None else ['audi', 'bmw', 'honda']
        brand = c1.selectbox("Marque", sorted(brand_options), index=0)
        model_year = c2.number_input("Année modèle", 1990, 2024, 2020, 1)

        c3, c4 = st.sidebar.columns(2)
        kilometers = c3.number_input("Kilométrage", 0, 500000, 50000, 1000)
        fuel_type = c4.selectbox("Carburant", ['gas', 'diesel'], index=0)

        c5, c6 = st.sidebar.columns(2)
        aspiration = c5.selectbox("Aspiration", ['std', 'turbo'], index=0)
        doornumber = c6.selectbox("Portes", ['two', 'four'], index=0)

        carbody = st.sidebar.selectbox("Carrosserie", ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'], index=0)
        drivewheel = st.sidebar.selectbox("Traction", ['fwd', 'rwd', '4wd'], index=0)

        st.sidebar.header("Spécifications")
        # Utilisation d'expander pour ne pas encombrer
        with st.sidebar.expander("Moteur & Performances", expanded=True):
            c7, c8 = st.columns(2)
            enginesize = c7.number_input("Taille moteur (cc)", 60, 500, 130, 10)
            horsepower = c8.number_input("Puissance (bhp)", 50, 300, 100, 5)

            c9, c10 = st.columns(2)
            curbweight = c9.number_input("Poids (kg)", 1500, 5000, 2500, 50)
            cylindernumber = c10.selectbox("Cylindres", ['four', 'six', 'five', 'eight', 'two', 'twelve', 'three'], index=0)

            c11, c12 = st.columns(2)
            citympg = c11.number_input("Ville (mpg)", 10, 50, 25, 1)
            highwaympg = c12.number_input("Autoroute (mpg)", 15, 60, 30, 1)

        # Créer les features dans l'ordre exact du modèle
        # ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'] + encoded cols

        # Valeurs par défaut pour les features non exposées dans l'interface
        symboling = 1  # valeur moyenne
        wheelbase = 98.0  # moyenne
        carlength = 175.0  # moyenne
        carwidth = 66.0  # moyenne
        carheight = 54.0  # moyenne
        boreratio = 3.3  # moyenne
        stroke = 3.3  # moyenne
        compressionratio = 9.0  # moyenne
        peakrpm = 5000  # valeur commune

        features = [
            symboling, wheelbase, carlength, carwidth, carheight, curbweight,
            enginesize, boreratio, stroke, compressionratio, horsepower,
            peakrpm, citympg, highwaympg
        ]

        # Ajouter les features encodées
        for col in categorical_cols:
            if col in encoders:
                if col == 'brand':
                    features.append(encoders[col].transform([brand])[0])
                elif col == 'fueltype':
                    features.append(encoders[col].transform([fuel_type])[0])
                elif col == 'aspiration':
                    features.append(encoders[col].transform([aspiration])[0])
                elif col == 'doornumber':
                    features.append(encoders[col].transform([doornumber])[0])
                elif col == 'carbody':
                    features.append(encoders[col].transform([carbody])[0])
                elif col == 'drivewheel':
                    features.append(encoders[col].transform([drivewheel])[0])
                elif col == 'enginelocation':
                    features.append(encoders[col].transform(['front'])[0])  # Default to front
                elif col == 'enginetype':
                    features.append(encoders[col].transform(['ohc'])[0])  # Default to ohc
                elif col == 'fuelsystem':
                    features.append(encoders[col].transform(['mpfi'])[0])  # Default to mpfi
                elif col == 'cylindernumber':
                    features.append(encoders[col].transform([cylindernumber])[0])  # Use user input
                else:
                    # Valeurs par défaut pour les autres colonnes catégorielles
                    features.append(encoders[col].transform([encoders[col].classes_[0]])[0])

        return features, {
            "brand": brand, "model_year": model_year, "kilometers": kilometers,
            "fuel_type": fuel_type, "aspiration": aspiration, "doornumber": doornumber,
            "carbody": carbody, "drivewheel": drivewheel, "enginesize": enginesize,
            "horsepower": horsepower, "curbweight": curbweight, "cylindernumber": cylindernumber,
            "citympg": citympg, "highwaympg": highwaympg
        }

    user_features_list, user_inputs_dict = get_user_input()

    # --- 4. DASHBOARD CENTRAL ---

    # Calculs
    predicted_price = predict_price_manual(user_features_list)

    # --- HEADER : RÉSULTAT CLÉ EN MAIN ---
    col_main, col_gauge = st.columns([2, 1])

    with col_main:
        st.subheader("Estimation de Valeur")
        st.markdown(f"<h1 style='font-size: 60px; color: #4CAF50;'>${predicted_price:,.0f}</h1>", unsafe_allow_html=True)
        st.caption(f"Pour ce {user_inputs_dict['brand']} {user_inputs_dict['model_year']} avec {user_inputs_dict['kilometers']:,} km.")

        # Préparation des fichiers
        # 1. CSV
        export_df = pd.DataFrame([user_inputs_dict])
        export_df['Estimated_Price'] = predicted_price
        csv_data = export_df.to_csv(index=False).encode('utf-8')

        # 2. PDF
        try:
            pdf_data = create_pdf_report(user_inputs_dict, predicted_price)
            pdf_available = True
        except Exception as e:
            pdf_available = False

        # Boutons côte à côte
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.download_button(
                label="Télécharger Rapport CSV",
                data=csv_data,
                file_name='estimation_automobile.csv',
                mime='text/csv',
                use_container_width=True
            )
        with btn_col2:
            if pdf_available:
                st.download_button(
                    label="Télécharger Rapport PDF",
                    data=pdf_data,
                    file_name='estimation_automobile.pdf',
                    mime='application/pdf',
                    use_container_width=True
                )
            else:
                st.warning("Installez 'fpdf' pour le PDF")

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
    tab1, tab2, tab3, tab4 = st.tabs(["Financement", "Dépréciation", "Comparables", "Analyse Avancée"])

    # === ONGLET 1 : FINANCEMENT ===
    with tab1:
        st.subheader("Analyse de Financement")
        st.markdown("Estimation des mensualités et coût total du crédit.")

        col_inv1, col_inv2 = st.columns(2)
        with col_inv1:
            down_payment_pct = st.slider("Apport initial (%)", 10, 50, 20, 5)
            loan_term = st.slider("Durée du prêt (ans)", 2, 7, 5)
            interest_rate = st.slider("Taux d'intérêt annuel (%)", 3.0, 12.0, 6.0, 0.5)

        with col_inv2:
            down_payment = predicted_price * (down_payment_pct / 100)
            loan_amount = predicted_price - down_payment
            monthly_rate = interest_rate / 100 / 12
            num_payments = loan_term * 12

            if monthly_rate > 0:
                monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            else:
                monthly_payment = loan_amount / num_payments

            total_cost = monthly_payment * num_payments + down_payment

            st.metric("Mensualité", f"${monthly_payment:,.0f}", delta="Par mois")
            st.metric("Coût Total", f"${total_cost:,.0f}",
                    delta=f"+${total_cost-predicted_price:,.0f} d'intérêt")

            st.caption(f"*Basé sur un apport de ${down_payment:,.0f} et {num_payments} mensualités.")
            if monthly_payment < predicted_price * 0.02:  # Less than 2% of car value
                st.success("Financement abordable.")
            else:
                st.warning("Mensualité élevée - considérer un apport plus important.")

    # === ONGLET 2 : SIMULATEUR DÉPRÉCIATION ===
    with tab2:
        st.subheader("Potentiel de Valorisation")
        scenarios = []

        # Impact of better horsepower
        if user_inputs_dict['horsepower'] < 150:
            feats_more_power = user_features_list.copy()
            feats_more_power[10] = min(user_inputs_dict['horsepower'] + 20, 200)  # More horsepower
            price_more_power = predict_price_manual(feats_more_power)
            scenarios.append({"Action": "+20 bhp de puissance", "Gain": price_more_power - predicted_price})

        # Impact of larger engine
        if user_inputs_dict['enginesize'] < 180:
            feats_bigger_engine = user_features_list.copy()
            feats_bigger_engine[6] = min(user_inputs_dict['enginesize'] + 30, 200)  # Bigger engine
            price_bigger_engine = predict_price_manual(feats_bigger_engine)
            scenarios.append({"Action": "+30cc de cylindrée", "Gain": price_bigger_engine - predicted_price})

        # Impact of turbo aspiration
        if user_inputs_dict['aspiration'] != 'turbo':
            feats_turbo = user_features_list.copy()
            # Find aspiration index and change to turbo
            for i, col in enumerate(categorical_cols):
                if col == 'aspiration':
                    base_idx = len(feature_columns) - len(categorical_cols)
                    feats_turbo[base_idx + i] = encoders['aspiration'].transform(['turbo'])[0]
                    break
            price_turbo = predict_price_manual(feats_turbo)
            scenarios.append({"Action": "Ajouter turbo", "Gain": price_turbo - predicted_price})

        # Impact of RWD drive
        if user_inputs_dict['drivewheel'] != 'rwd':
            feats_rwd = user_features_list.copy()
            # Find drivewheel index and change to rwd
            for i, col in enumerate(categorical_cols):
                if col == 'drivewheel':
                    base_idx = len(feature_columns) - len(categorical_cols)
                    feats_rwd[base_idx + i] = encoders['drivewheel'].transform(['rwd'])[0]
                    break
            price_rwd = predict_price_manual(feats_rwd)
            scenarios.append({"Action": "Traction arrière (RWD)", "Gain": price_rwd - predicted_price})

        if not scenarios:
            st.info("Ce véhicule est déjà très bien positionné !")
        else:
            max_gain = max([s['Gain'] for s in scenarios]) if scenarios else 1.0
            if max_gain == 0: max_gain = 1.0

            cols_sim = st.columns(len(scenarios))
            for i, scen in enumerate(scenarios):
                with cols_sim[i]:
                    gain_value = scen['Gain']
                    if gain_value > 0:
                        st.metric(label=scen["Action"], value=f"+ ${gain_value:,.0f}", delta="Plus-value")
                        st.progress(min(1.0, gain_value / (max_gain * 1.2)))
                    else:
                        st.metric(label=scen["Action"], value=f"${gain_value:,.0f}", delta="Moins-value")
                        # Don't show progress bar for negative values

    # === ONGLET 3 : COMPARABLES ===
    with tab3:
        st.subheader("Véhicules Similaires Vendus")
        similar_df = get_similar_cars(user_features_list, df_raw)
        st.dataframe(
            similar_df[['price', 'CarName', 'fueltype', 'carbody', 'enginesize', 'horsepower']],
            column_config={
                "price": st.column_config.NumberColumn("Prix", format="$%d"),
                "CarName": st.column_config.TextColumn("Modèle"),
                "fueltype": "Carburant",
                "carbody": "Carrosserie",
                "enginesize": st.column_config.NumberColumn("Moteur", format="%d cc"),
                "horsepower": st.column_config.NumberColumn("Puissance", format="%d bhp")
            },
            use_container_width=True,
            hide_index=True
        )
        avg_sim_price = similar_df['price'].mean()
        st.info(f"Prix moyen des comparables : ${avg_sim_price:,.0f}")

    # === ONGLET 4 : ANALYSE AVANCÉE ===
    with tab4:
        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.subheader("Sensibilité Prix / Kilométrage")
            st.caption("Comment évolue le prix de VOTRE configuration si le kilométrage change ?")

            horsepower_range = np.linspace(50, 200, 50)
            prices_sensitivity = []
            base_features = list(user_features_list)
            for hp in horsepower_range:
                temp_feats = base_features.copy()
                temp_feats[10] = hp  # Index 10 is horsepower in our feature array
                prices_sensitivity.append(predict_price_manual(temp_feats))

            fig_sens = px.line(x=horsepower_range, y=prices_sensitivity, labels={'x': 'Puissance (bhp)', 'y': 'Prix Estimé ($)'})
            fig_sens.add_trace(go.Scatter(x=[user_inputs_dict['horsepower']], y=[predicted_price], mode='markers', marker=dict(color='red', size=10), name="Actuel"))
            st.plotly_chart(fig_sens, use_container_width=True)

        with col_a2:
            # Option 1: Positionnement
            st.subheader("Positionnement sur le Marché")
            st.caption("Où se situe votre véhicule par rapport à l'offre existante ?")
            fig_dist = px.histogram(df_raw, x="price", nbins=30, color_discrete_sequence=['#cbd5e1'], opacity=0.7, labels={'price': 'Prix ($)'})
            fig_dist.add_vline(x=predicted_price, line_width=3, line_dash="dash", line_color="#4CAF50", annotation_text="Votre estimation", annotation_position="top right")
            fig_dist.update_layout(xaxis_title="Prix ($)", yaxis_title="Nombre de véhicules", showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_dist, use_container_width=True)
            percentile = (df_raw['price'] < predicted_price).mean() * 100
            st.info(f"Ce véhicule est plus cher que **{percentile:.0f}%** des voitures du marché.")

        # Option 2: Radar
        st.markdown("---")
        st.subheader("Profil du Véhicule vs Moyenne (Radar)")
        categories = ['enginesize', 'horsepower', 'curbweight', 'citympg', 'highwaympg']
        user_vals = []
        avg_vals = []

        # Mapping from dataset column names to user input keys
        key_mapping = {
            'enginesize': 'enginesize',
            'horsepower': 'horsepower',
            'curbweight': 'curbweight',
            'citympg': 'citympg',
            'highwaympg': 'highwaympg'
        }

        for cat in categories:
            max_val = df_raw[cat].max()
            user_key = key_mapping[cat]
            user_vals.append(user_inputs_dict[user_key] / max_val)
            avg_vals.append(df_raw[cat].mean() / max_val)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=user_vals, theta=categories, fill='toself', name='Votre Véhicule', line_color='#4CAF50'))
        fig_radar.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='Moyenne Marché', line_color='#94a3b8', opacity=0.5))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), margin=dict(l=40, r=40, t=30, b=30), showlegend=True, height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

        # Option 3: Scatter Plot
        st.markdown("---")
        st.subheader("Ratio Prix / Puissance")
        st.caption("Comparaison avec les transactions réelles.")
        fig_scat = px.scatter(df_raw, x="horsepower", y="price", color_discrete_sequence=['#cbd5e1'], opacity=0.6, labels={'horsepower': 'Puissance (bhp)', 'price': 'Prix ($)'})
        fig_scat.add_trace(go.Scatter(x=[user_inputs_dict['horsepower']], y=[predicted_price], mode='markers', marker=dict(color='#4CAF50', size=15, symbol='star'), name="Votre Véhicule"))
        fig_scat.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scat, use_container_width=True)