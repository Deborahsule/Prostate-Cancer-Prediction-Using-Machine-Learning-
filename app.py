import streamlit as st
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

# Define the RNNModel class
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Load the trained models
rnn_model = torch.load('rnn_model.pth')


# Load the scaler
scaler = joblib.load('scaler.sav')

# Define the RNN prediction function
def predict_with_rnn(input_data):
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
    input_tensor = torch.FloatTensor(scaler.transform(input_tensor))
    input_tensor = input_tensor.unsqueeze(0)  # Add sequence dimension
    rnn_model.eval()
    with torch.no_grad():
        output = rnn_model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Define the Random Forest prediction function
def predict_with_random_forest(input_data):
    input_scaled = scaler.transform([input_data])
    prediction = random_forest_model.predict(input_scaled)
    return prediction[0]

# Streamlit app
st.title("Prostate Cancer Prediction")

st.write("Input the features to predict the target class using RNN.")

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
    rnn_prediction = predict_with_rnn(input_data)
    
    
    st.write(f"RNN Prediction: {rnn_prediction}")
    