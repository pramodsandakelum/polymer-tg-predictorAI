import streamlit as st
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
import joblib
import py3Dmol

# Load your trained stacked model and scaler once
@st.cache_resource(show_spinner=True)
def load_model_and_scaler():
    full_model = joblib.load('full_stacked_model.pkl')
    feature_scaler = joblib.load('feature_scaler.pkl')
    return full_model, feature_scaler

full_model, feature_scaler = load_model_and_scaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Featurization functions
def featurize_combo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
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
        return None
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

def get_features(smiles):
    fps = featurize_combo(smiles)
    desc = calc_extended_descriptors(smiles)
    if fps is None or desc is None:
        return None
    features = np.hstack([fps, desc]).reshape(1, -1)
    return features

def render_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(mol)
    block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=350)
    viewer.addModel(block, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    st.components.v1.html(viewer.show(), height=360, width=400)

# Streamlit UI
st.title("🔬 Polymer Property Predictor from SMILES")
st.write("Enter a **single** polymer SMILES string below and get predicted properties:")

smiles_input = st.text_input("📥 Input Polymer SMILES", "")

if smiles_input.strip():
    features = get_features(smiles_input)
    if features is None:
        st.error("❌ Invalid SMILES string. Please enter a valid polymer SMILES.")
    else:
        # Show 3D molecular structure
        st.write("### 🧪 3D Molecular Structure")
        render_molecule(smiles_input)

        # Scale features
        X_scaled = feature_scaler.transform(features)

        # Predict with full stacked model
        preds = full_model.predict(features)
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        pred_dict = {target: float(preds[0, i]) for i, target in enumerate(targets)}

        st.write("### 📊 Predicted Polymer Properties")
        st.table(pd.DataFrame(pred_dict, index=[0]))
else:
    st.info("ℹ️ Please enter a polymer SMILES string to get started.")
