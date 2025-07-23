import streamlit as st
import joblib
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
import py3Dmol
import pandas as pd

# Featurization functions (same as training)
def featurize_combo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048 + 167)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    morgan_arr = np.zeros((2048,), dtype=int)
    DataStructs.ConvertToNumpyArray(morgan_fp, morgan_arr)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros((167,), dtype=int)
    DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr)
    return np.concatenate([morgan_arr, maccs_arr])

def calc_extended_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(12)
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumValenceElectrons(mol),
    ]
    return np.array(desc)

# Your NN model class from training
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dims=[512, 128], dropout_rates=[0.3, 0.2]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim, drop in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Full stacked model wrapper
class FullStackedModel:
    def __init__(self, models, nn_models, stack_model, feature_scaler, targets):
        self.models = models
        self.nn_models = nn_models
        self.stack_model = stack_model
        self.feature_scaler = feature_scaler
        self.targets = targets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, X_raw):
        X_scaled = self.feature_scaler.transform(X_raw)
        preds_list = []
        for target in self.targets:
            model = self.models[target]
            model_nn = self.nn_models[target]

            lgb_pred = model.predict(X_scaled)

            model_nn.eval()
            with torch.no_grad():
                nn_pred = model_nn(torch.tensor(X_scaled, dtype=torch.float32).to(self.device)).cpu().numpy().reshape(-1)

            preds_avg = (lgb_pred + nn_pred) / 2
            preds_list.append(preds_avg)

        preds_stack_input = np.vstack(preds_list).T  # shape (n_samples, n_targets)
        final_preds = self.stack_model.predict(preds_stack_input)
        return final_preds

# Load models and scaler (make sure files are in working dir)
feature_scaler = joblib.load('feature_scaler.pkl')
full_model = joblib.load('full_stacked_model.pkl')
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Styling CSS
st.markdown(
    """
    <style>
    .viewer-wrapper {
        border: 3px solid #3498db;
        border-radius: 12px;
        padding: 5px;
        max-width: 450px;
        margin-bottom: 1rem;
        margin-left: auto;
        margin-right: auto;
    }
    table.dataframe {
        margin: 0 auto;
        border-collapse: collapse;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        width: 100%;
    }
    table.dataframe th, table.dataframe td {
        border: 1px solid #ddd !important;
        padding: 12px !important;
        text-align: center !important;
    }
    table.dataframe thead {
        background-color: #2980b9 !important;
        color: white !important;
    }
    tbody tr:hover {
        background-color: #f1f1f1 !important;
    }
    .pred-value {
        font-weight: 600;
        color: #27ae60;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Polymer Property Predictor with 3D Viewer 🔬")

col1, col2 = st.columns(2)

with col1:
    smiles_input = st.text_input("Enter a polymer SMILES string:", placeholder="e.g. C(C(=O)O)N")

    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("Invalid SMILES: Could not parse molecule.")
        else:
            mb = Chem.MolToMolBlock(mol)
            viewer = py3Dmol.view(width=400, height=350)
            viewer.addModel(mb, 'mol')
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            html = viewer._make_html()

            # Wrap the 3D viewer in a bordered div container
            
            st.components.v1.html(html, height=370)
            

with col2:
    if smiles_input:
        fps = featurize_combo(smiles_input)
        desc = calc_extended_descriptors(smiles_input)
        features = np.hstack([fps, desc]).reshape(1, -1)
        preds = full_model.predict(features)

        # Prepare DataFrame for nicer display
        df_preds = pd.DataFrame({
            "Property": targets,
            "Predicted Value": [f"{v:.4f}" for v in preds[0]]
        })

        # Style the dataframe for border & color on predicted values
        def highlight_pred(val):
            return 'color: #27ae60; font-weight: 600'

        st.dataframe(df_preds.style.applymap(highlight_pred, subset=["Predicted Value"]))

# Footer
st.markdown(
    """
    <div style="text-align:center; margin-top: 3rem; color: #95a5a6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    Made with ❤️ by Pramod • Powered by Streamlit & RDKit
    </div>
    """,
    unsafe_allow_html=True,
)
