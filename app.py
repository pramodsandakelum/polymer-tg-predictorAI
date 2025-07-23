import streamlit as st
import joblib
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
import py3Dmol

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

# Your NN model class from training (omitted for brevity)

# Full stacked model wrapper class (omitted for brevity)

# Load models and scaler (make sure files are in working dir)
feature_scaler = joblib.load('feature_scaler.pkl')
full_model = joblib.load('full_stacked_model.pkl')
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

st.title("Polymer Property Predictor with 3D Viewer 🔬")

col1, col2 = st.columns([1,1])

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

            # Wrap the 3D viewer with a visible border and some padding:
            bordered_html = f"""
            <div style="
                border: 3px solid #3498db; 
                border-radius: 12px; 
                padding: 8px; 
                max-width: 420px; 
                margin: auto;">
                {html}
            </div>
            """
            st.components.v1.html(bordered_html, height=370)

with col2:
    if 'smiles_input' in locals() and smiles_input:
        fps = featurize_combo(smiles_input)
        desc = calc_extended_descriptors(smiles_input)
        features = np.hstack([fps, desc]).reshape(1, -1)
        preds = full_model.predict(features)

        # Build a bordered HTML table string with inline CSS for styling
        table_html = """
        <table style="
            border-collapse: collapse; 
            width: 100%; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <thead>
                <tr style="background-color: #2980b9; color: white;">
                    <th style="border: 1px solid #ddd; padding: 8px;">Property</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Predicted Value</th>
                </tr>
            </thead>
            <tbody>
        """

        for i, target in enumerate(targets):
            table_html += f"""
                <tr style="text-align: center;">
                    <td style="border: 1px solid #ddd; padding: 8px;">{target}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: 600; color: #27ae60;">{preds[0, i]:.4f}</td>
                </tr>
            """

        table_html += "</tbody></table>"

        st.markdown(table_html, unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <div style="text-align:center; margin-top: 3rem; color: #95a5a6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    Made with ❤️ by Pramod • Powered by Streamlit & RDKit
    </div>
    """,
    unsafe_allow_html=True,
)
