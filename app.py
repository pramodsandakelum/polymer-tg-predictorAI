import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
import py3Dmol

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- FullStackedModel class --------
class FullStackedModel:
    def __init__(self, models, nn_models, stack_model, feature_scaler, targets):
        self.models = models
        self.nn_models = nn_models
        self.stack_model = stack_model
        self.feature_scaler = feature_scaler
        self.targets = targets
        self.device = device

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

# -------- Featurization functions --------
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

# -------- Load model and scaler --------
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model = joblib.load("full_stacked_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# -------- Streamlit UI --------
st.title("Polymer Property Predictor")
st.write("Enter a polymer SMILES string to predict Tg, FFV, Tc, Density, and Rg.")

smiles_input = st.text_input("Polymer SMILES:", value="CCO")

if st.button("Predict"):

    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("Invalid SMILES string! Please enter a valid molecule.")
    else:
        # Featurize input
        fps = featurize_combo(smiles_input)
        desc = calc_extended_descriptors(smiles_input)
        X_raw = np.hstack([fps, desc]).reshape(1, -1)

        # Predict
        preds = model.predict(X_raw)

        # Display results
        results_df = pd.DataFrame(preds, columns=targets).T
        results_df.columns = ["Predicted Value"]
        st.table(results_df.style.format("{:.4f}"))

        # Show 3D molecule
        mblock = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=400, height=300)
        viewer.addModel(mblock, "mol")
        viewer.setStyle({"stick": {}})
        viewer.setBackgroundColor('0xeeeeee')
        viewer.zoomTo()
        viewer.show()
        st.components.v1.html(viewer.js(), height=350)

