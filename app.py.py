# Databricks notebook source
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the saved artifacts
@st.cache_resource  # Cache for performance
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('le.pkl')
    ohe = joblib.load('ohe.pkl')
    return model, scaler, le, ohe

model, scaler, le, ohe = load_artifacts()

# Define features (match your notebook)
numerical_features = ['Throttle', 'Brake', 'Speed', 'Surface_Roughness',
                      'Ambient_Temperature', 'Lateral_G_Force', 'Longitudinal_G_Force',
                      'Tire_Friction_Coefficient', 'Tire_Tread_Depth',
                      'force_on_tire', 'front_surface_temp', 'rear_surface_temp',
                      'front_inner_temp', 'rear_inner_temp']
categorical_features = ['Tire_Compound', 'Driving_Style', 'Track']

# Streamlit UI
st.title("F1 Tire Degradation Risk Predictor")
st.markdown("""
Enter current telemetry data to predict tire degradation risk. 
This helps teams decide pit stops—e.g., if 'critical', pit immediately to avoid 1-2s/lap loss (FIA mandates safe tires per Article 28).
""")

# Input sections
st.header("Numerical Telemetry Inputs")
num_inputs = {}
for feature in numerical_features:
    if feature == 'Throttle' or feature == 'Brake':
        num_inputs[feature] = st.slider(f"{feature} (%)", 0.0, 100.0, 50.0)
    elif feature == 'Speed':
        num_inputs[feature] = st.slider(f"{feature} (km/h)", 0.0, 350.0, 200.0)
    elif 'G_Force' in feature:
        num_inputs[feature] = st.slider(f"{feature} (G)", -5.0, 5.0, 0.0)
    elif 'temp' in feature:
        num_inputs[feature] = st.slider(f"{feature} (°C)", 0.0, 150.0, 80.0)
    elif feature == 'Tire_Tread_Depth':
        num_inputs[feature] = st.slider(f"{feature} (mm)", 0.0, 10.0, 5.0)
    else:
        num_inputs[feature] = st.number_input(f"{feature}", value=0.0)

st.header("Categorical Inputs")
cat_inputs = {}
cat_inputs['Tire_Compound'] = st.selectbox("Tire Compound", ['Soft', 'Medium', 'Hard'])  # Adjust options based on data
cat_inputs['Driving_Style'] = st.selectbox("Driving Style", ['Aggressive', 'Balanced', 'Conservative'])  # Adjust
cat_inputs['Track'] = st.selectbox("Track", ['Monza', 'Monaco', 'Red Bull Ring'])

# Predict button
if st.button("Predict Degradation Risk"):
    # Prepare input DataFrame
    input_df_num = pd.DataFrame([num_inputs])
    input_df_cat = pd.DataFrame([cat_inputs])
    
    # Encode categoricals
    cat_encoded = ohe.transform(input_df_cat)
    
    # Combine and scale
    input_combined = np.hstack((input_df_num, cat_encoded))
    input_scaled = scaler.transform(input_combined)
    
    # Predict
    pred = model.predict(input_scaled)[0]
    risk_class = le.inverse_transform([pred])[0]
    
    # Output with F1 insight
    st.success(f"Predicted Risk: **{risk_class.upper()}**")
    if risk_class == 'safe':
        st.info("Tires good—push harder. In F1, this means extend stint, gain positions (e.g., undercut strategy).")
    elif risk_class == 'medium':
        st.warning("Monitor closely—plan pit soon. Degradation rising; could add 0.5-1s/lap if ignored.")
    else:
        st.error("Critical—pit now! Risks DNF or penalty (FIA Article 28.4). Switch compounds to optimize.")

# Footer
st.markdown("---\nBuilt for CIS 508 Final Project. Model: XGBoost Classifier.")
