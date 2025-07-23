import streamlit as st
import numpy as np
import torch
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
from stmol import showmol
import py3Dmol

# Load model and scaler
model = joblib.load("full_stacked_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

# Featurization functions
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

def featurize(smiles):
    return np.hstack([featurize_combo(smiles), calc_extended_descriptors(smiles)])

def render_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=350)
    viewer.addModel(block, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    showmol(viewer, height=350, width=400)

# Streamlit UI
st.set_page_config(layout="centered", page_title="Polymer Property Predictor")
st.title("🔬 Polymer Property Predictor with 3D Viewer")
st.markdown("Predicts **Tg**, **FFV**, **Tc**, **Density**, and **Rg** from a single SMILES input using a stacked ensemble model.")

smiles_input = st.text_input("📥 Enter a SMILES string", "")

if st.button("Predict") and smiles_input.strip():
    with st.spinner("Featurizing and predicting..."):
        features = featurize(smiles_input).reshape(1, -1)
        prediction = model.predict(features)[0]
        targets = model.targets

        st.subheader("📊 Predicted Polymer Properties")
        for i, t in enumerate(targets):
            st.write(f"**{t}**: `{prediction[i]:.2f}`")

        st.subheader("🧪 3D Molecular Structure")
        render_molecule(smiles_input)
