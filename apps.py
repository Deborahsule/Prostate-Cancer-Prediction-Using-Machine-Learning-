import streamlit as st
import numpy as np
import joblib

# Load the saved models
scaler = joblib.load('scaler.sav')
pca = joblib.load('pca.sav')
rf_model = joblib.load('rf_model.sav')

# Define the prediction function
def predict_rf(input_data):
    # Convert input data to numpy array
    input_array = np.array(input_data).reshape(1, -1)
    
    # Scale the features
    scaled = scaler.transform(input_array)

    # Apply PCA transformation
    pca_features = pca.transform(scaled)
    
    
    
    # Make the prediction
    prediction = rf_model.predict(pca_features)
    return prediction[0]

# Streamlit app
st.title("Prostate Cancer Prediction Using Random Forest Classifier")

st.write("Input the features to predict the target class using the Random Forest Classifier.")

# Input fields
lcavol = st.number_input('lcavol', value=0.0)
lweight = st.number_input('lweight', value=0.0)
age = st.number_input('age', value=50)
lbph = st.number_input('lbph', value=0.0)
lcp = st.number_input('lcp', value=0.0)
gleason = st.number_input('gleason', value=0.0)
pgg45 = st.number_input('pgg45', value=0.0)
lpsa = st.number_input('lpsa', value=0.0)

input_data = [lcavol, lweight, age, lbph, lcp, gleason, pgg45, lpsa]

# Predict button
if st.button('Predict'):
    prediction = predict_rf(input_data)
    st.write(f"Random Forest Prediction: {prediction}")
