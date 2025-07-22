import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
import joblib
import torch
import torch.nn as nn
import py3Dmol

# Device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the SimpleNN class exactly as in your training code
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# Define your StackedModel class as well, if you use it to load
class StackedModel:
    def __init__(self, model_lgb, model_nn, stack_model, scaler, input_dim):
        self.model_lgb = model_lgb
        self.model_nn = model_nn
        self.stack_model = stack_model
        self.scaler = scaler
        self.input_dim = input_dim

    def predict(self, X_np):
        X_scaled = self.scaler.transform(X_np)
        lgb_pred = self.model_lgb.predict(X_scaled)
        self.model_nn.eval()
        with torch.no_grad():
            nn_pred = self.model_nn(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)
        stacked_input = np.vstack([lgb_pred, nn_pred]).T
        return self.stack_model.predict(stacked_input)

# Now load your stacked model and scaler
model = joblib.load('stacked_polymer_model.pkl')
scaler = joblib.load('polymer_feature_scaler.pkl')

# (Rest of your app code here: featurization, UI, prediction, 3D viewer...)

# Example of fingerprint function
def featurize_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    morgan_arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(morgan_fp, morgan_arr)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros((167,), dtype=int)
    DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr)
    return np.concatenate([morgan_arr, maccs_arr])

# 3D viewer function, main app function, etc...
# Make sure to keep your app code consistent with this model loading

# Example main function
def main():
    st.title("Polymer Tg Prediction App")
    
    st.markdown("""
    Welcome to the **Polymer Tg Prediction App**  
    This tool helps you **predict the glass transition temperature (Tg)** of a polymer using its **SMILES structure**.  
    You’ll also get a **3D visualization** of your polymer structure.

    ### What is Tg (Glass Transition Temperature)?

    The **glass transition temperature (Tg)** is the temperature at which a polymer changes from a **hard and brittle "glassy" state** to a **soft and flexible "rubbery" state**.  
    It's a critical property for determining a polymer's **mechanical behavior**, **flexibility**, and **temperature resistance** in real-world applications.

    ### How to Use:
    1. Enter a valid **SMILES string** of the polymer.
    2. Click **Predict Tg and Show 3D** to get:
        - Estimated **Tg value**
        - **Interactive 3D model** of your molecule
    """)

    smiles = st.text_input("Enter Polymer SMILES:")
    if st.button("Predict Tg"):
        if not smiles:
            st.error("Please enter a SMILES string.")
            return

        features = featurize_fp(smiles)
        if features is None:
            st.error("Invalid SMILES or failed fingerprint generation.")
            return

        features = features.reshape(1, -1)
        try:
            prediction = model.predict(features)
            st.success(f"Predicted Tg: {prediction[0]:.2f} K")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
