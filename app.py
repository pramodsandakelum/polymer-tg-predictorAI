import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
import xgboost as xgb
import joblib
import py3Dmol

# Load model and scaler
model = joblib.load('stacked_polymer_model.pkl')
scaler = joblib.load('polymer_feature_scaler.pkl')

# Combined fingerprint function
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

# 3D viewer function
def draw_3d_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        st.warning(f"3D embedding failed: {e}")
        return None
    mb = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mb, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    return viewer

# Main Streamlit app
def main():
    st.title("Polymer Tg Prediction + 3D Viewer")
    
    # App Introduction
    st.markdown("""
    Welcome to the **Polymer Tg Prediction App
    This tool helps you **predict the glass transition temperature (Tg)** of a polymer using its **SMILES structure** and selected molecular properties.  
    You’ll also get a **3D visualization** of your polymer structure.

    ###  What is Tg (Glass Transition Temperature)?

    The **glass transition temperature (Tg)** is the temperature at which a polymer changes from a **hard and brittle "glassy" state** to a **soft and flexible "rubbery" state**.  
    It's a critical property for determining a polymer's **mechanical behavior**, **flexibility**, and **temperature resistance** in real-world applications.

    ###  How to Use:
    1. Enter a valid **SMILES string** of the polymer.
    2. Input optional molecular features like:
        - FFV (Free Volume Fraction)
        - Tc (Critical Temperature)
        - Density (g/cm³)
        - Rg (Radius of Gyration)
        - Molecular Weight
    3. Click **Predict Tg and Show 3D** to get:
        - Estimated **Tg value**
        - **Interactive 3D model** of your molecule
    """)
    # User input for SMILES
    smiles = st.text_input("Enter Polymer SMILES:")

    # Extra features
    st.subheader("Enter Additional Molecular Properties:")
    ffv = st.number_input("FFV (Free Volume Fraction)", min_value=0.0, step=0.01, format="%.4f")
    tc = st.number_input("Tc (Critical Temperature in K)", min_value=0.0, step=1.0)
    density = st.number_input("Density (g/cm³)", min_value=0.0, step=0.01, format="%.4f")
    rg = st.number_input("Rg (Radius of Gyration)", min_value=0.0, step=0.1)
    mw = st.number_input("Molecular Weight", min_value=0.0, step=1.0)

    if st.button("Predict Tg and Show 3D"):
        if not smiles:
            st.error("Please enter a SMILES string.")
            return

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES string.")
            return

        st.write(f"Input SMILES: `{smiles}`")

        with st.spinner("Predicting Tg and generating 3D structure..."):
            # Generate features
            features = featurize_fp(smiles)
            if features is None:
                st.error("Failed to generate molecular fingerprints.")
                return

            # Append additional features
            extra_features = [ffv, tc, density, rg, mw]
            full_features = np.concatenate([features, extra_features])

            # Scale and predict
            try:
                features_scaled = scaler.transform([full_features])
                dmatrix = xgb.DMatrix(features_scaled)
                pred = model.predict(dmatrix)
                st.success(f"Predicted Tg: {pred[0]:.2f} K")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            # 3D Structure
            st.subheader("3D Polymer Structure")
            viewer = draw_3d_molecule(smiles)
            if viewer:
                viewer_html = viewer._make_html()
                st.components.v1.html(viewer_html, height=450, width=450)
            else:
                st.info("3D structure not available for this molecule.")

if __name__ == "__main__":
    main()
