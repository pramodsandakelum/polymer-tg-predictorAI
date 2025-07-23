import streamlit as st
import joblib
import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs

# Load model and scaler
model = joblib.load("full_stacked_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

# Define molecular descriptors
def compute_descriptors(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol)
    ]

# Combine all features
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol, clearAromaticFlags=True)

    # Morgan fingerprint
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    morgan_fp = np.array(morgan_fp)

    # MACCS keys
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_fp = np.array(maccs_fp)[1:]  # skip first bit

    # Molecular descriptors
    desc = compute_descriptors(mol)

    # Concatenate all
    features = np.concatenate([morgan_fp, maccs_fp, desc])
    return features.reshape(1, -1)

# Display molecule in 3D
def show_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mb, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    st.components.v1.html(viewer._make_html(), height=450)

# Streamlit UI
st.set_page_config(page_title="Polymer Property Predictor", layout="centered")
st.title("🧪 Polymer Property Predictor")
st.markdown("Enter a polymer SMILES string below to predict **Tg**, **FFV**, **Tc**, **Density**, and **Rg**.")

smiles = st.text_input("📥 Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")

if st.button("🔍 Predict Properties"):
    with st.spinner("Processing..."):
        features = featurize(smiles)
        if features is None:
            st.error("Invalid SMILES. Please check your input.")
        else:
            scaled = scaler.transform(features)
            predictions = model.predict(scaled)[0]

            # Show predictions
            props = ["Tg (K)", "FFV", "Tc (K)", "Density (g/cm³)", "Rg (Å)"]
            st.subheader("🔬 Predicted Polymer Properties")
            for p, v in zip(props, predictions):
                st.write(f"**{p}**: {v:.2f}")

            # Show 3D molecule
            st.subheader("🧬 3D Molecular Structure")
            show_3d(smiles)
