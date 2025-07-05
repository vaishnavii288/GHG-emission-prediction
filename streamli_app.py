import streamlit as st
import joblib
import numpy as np
import pandas as pd
from process import preprocess_input

st.markdown(
    """
    <style>
    .stApp {
       background-image: url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");

        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        min-height: 100vh;
    }
    .main-title, .main-desc {
        color: black !important;
        text-align: left;
    }
    .stNumberInput label, .stSlider label, .stSelectbox label {
        color: black !important;
        font-weight: 10000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-title">GHG Emission Estimator for Sustainable Supply Chains</h1>', unsafe_allow_html=True)
st.markdown(
    '<h4 class="main-desc">Smart Estimation of Supply Chain Emissions Based on Key Data Quality Metrics.</h4>',
    unsafe_allow_html=True
)

model = joblib.load('models/LR_model.pkl')

scaler = joblib.load('models/scaler.pkl')

with st.form("prediction_form"):
    substance = st.selectbox("Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
    unit = st.selectbox("Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
    source = st.selectbox("Source", ['Commodity', 'Industry'])
    supply_wo_margin = st.number_input("Supply Chain Emission Factors without Margins", min_value=0.0)
    margin = st.number_input("Margins of Supply Chain Emission Factors", min_value=0.0)
    dq_reliability = st.slider("DQ Reliability", 0.0, 1.0)
    dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 1.0)
    dq_geo = st.slider("DQ Geographical Correlation", 0.0, 1.0)
    dq_tech = st.slider("DQ Technological Correlation", 0.0, 1.0)
    dq_data = st.slider("DQ Data Collection", 0.0, 1.0)
    # year = st.selectbox("Year", list(range(2010, 2017)))

    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        'Substance': substance,
        'Unit': unit,
        'Supply Chain Emission Factors without Margins': supply_wo_margin,
        'Margins of Supply Chain Emission Factors': margin,
        'DQ ReliabilityScore of Factors without Margins': dq_reliability,
        'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
        'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
        'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
        'DQ DataCollection of Factors without Margins': dq_data,
        'Source': source,
        # 'Year': year
    }

    input_df = preprocess_input(pd.DataFrame([input_data]))
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Supply Chain Emission Factor with Margin: **{prediction[0]:.4f}**")