import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd

session = get_active_session()

st.set_page_config(page_title="Prédiction Prix Immobilier", layout="wide")
st.title("Prédiction des Prix Immobiliers")
st.markdown("Utilisez ce formulaire pour estimer le prix d'une maison en fonction de ses caractéristiques.")

from snowflake.ml.registry import Registry
reg = Registry(session=session, database_name='HOUSE_PRICE', schema_name='ML')
model_ref = reg.get_model('house_price_predictor')
mv = model_ref.version('v1')

df_train = session.table('HOUSE_PRICE.ML.HOUSE_PRICES').to_pandas()

from sklearn.preprocessing import LabelEncoder, StandardScaler
binary_cols = ['ROUTE_PRINCIPALE', 'CHAMBRE_AMIS', 'SOUS_SOL', 'CHAUFFAGE_EAU_CHAUDE',
               'CLIMATISATION', 'ZONE_PRIVILEGIEE']
df_enc = df_train.copy()
for col in binary_cols:
    df_enc[col] = df_enc[col].map({'yes': 1, 'no': 0})
le = LabelEncoder()
df_enc['STATUT_AMEUBLEMENT_ENC'] = le.fit_transform(df_enc['STATUT_AMEUBLEMENT'])
feature_cols = ['SURFACE', 'CHAMBRES', 'SALLES_DE_BAIN', 'ETAGES', 'ROUTE_PRINCIPALE',
                'CHAMBRE_AMIS', 'SOUS_SOL', 'CHAUFFAGE_EAU_CHAUDE', 'CLIMATISATION',
                'PARKING', 'ZONE_PRIVILEGIEE', 'STATUT_AMEUBLEMENT_ENC']
scaler = StandardScaler()
scaler.fit(df_enc[feature_cols])

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Caractéristiques principales")
    surface = st.slider("Surface (m²)", min_value=10, max_value=500, value=100, step=5)
    chambres = st.selectbox("Nombre de chambres", options=[1, 2, 3, 4, 5, 6], index=2)
    salles_de_bain = st.selectbox("Nombre de salles de bain", options=[1, 2, 3, 4], index=0)
    etages = st.selectbox("Nombre d'étages", options=[1, 2, 3, 4], index=0)
    parking = st.selectbox("Places de parking", options=[0, 1, 2, 3], index=0)

with col2:
    st.subheader("Équipements")
    route_principale = st.radio("Route principale", options=["Oui", "Non"], index=0)
    chambre_amis = st.radio("Chambre d'amis", options=["Oui", "Non"], index=1)
    sous_sol = st.radio("Sous-sol", options=["Oui", "Non"], index=1)

with col3:
    st.subheader("Confort et emplacement")
    chauffage = st.radio("Chauffage eau chaude", options=["Oui", "Non"], index=1)
    clim = st.radio("Climatisation", options=["Oui", "Non"], index=1)
    zone_priv = st.radio("Zone privilégiée", options=["Oui", "Non"], index=1)
    ameublement = st.selectbox("Statut d'ameublement",
                               options=["furnished", "semi-furnished", "unfurnished"], index=1)

st.markdown("---")

if st.button("Prédire le prix", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        'SURFACE': surface,
        'CHAMBRES': chambres,
        'SALLES_DE_BAIN': salles_de_bain,
        'ETAGES': etages,
        'ROUTE_PRINCIPALE': 1 if route_principale == "Oui" else 0,
        'CHAMBRE_AMIS': 1 if chambre_amis == "Oui" else 0,
        'SOUS_SOL': 1 if sous_sol == "Oui" else 0,
        'CHAUFFAGE_EAU_CHAUDE': 1 if chauffage == "Oui" else 0,
        'CLIMATISATION': 1 if clim == "Oui" else 0,
        'PARKING': parking,
        'ZONE_PRIVILEGIEE': 1 if zone_priv == "Oui" else 0,
        'STATUT_AMEUBLEMENT_ENC': le.transform([ameublement])[0]
    }])

    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_cols)
    prediction = mv.run(input_scaled, function_name='predict')
    prix_predit = prediction['output_feature_0'].values[0]

    st.markdown("---")
    st.success(f"Prix estimé : **{prix_predit:,.0f}**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Prix estimé", f"{prix_predit:,.0f}")
    with col_b:
        prix_moyen = df_train['PRIX'].mean()
        diff = ((prix_predit - prix_moyen) / prix_moyen) * 100
        st.metric("Comparaison au prix moyen", f"{diff:+.1f}%")

    st.markdown("### Caractéristiques saisies")
    st.dataframe(input_data, use_container_width=True)
