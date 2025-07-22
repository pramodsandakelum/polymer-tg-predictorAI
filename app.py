import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs, Descriptors
import joblib
import torch
import py3Dmol

# Device for PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the StackedModel class exactly as during training
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

# Load the stacked model and scaler
model = joblib.load('stacked_polymer_model.pkl')
scaler = joblib.load('polymer_feature_scaler.pkl')

# Fingerprint function (Morgan + MACCS)
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

# 12 molecular descriptors
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
            features_fp = featurize_fp(smiles)
            if features_fp is None:
                st.error("Failed to generate molecular fingerprints.")
                return
            features_desc = calc_extended_descriptors(smiles)

            full_features = np.concatenate([features_fp, features_desc])

            try:
                features_scaled = scaler.transform([full_features])
            except Exception as e:
                st.error(f"Feature scaling failed: {e}")
                return

            try:
                pred = model.predict(features_scaled)
                st.success(f"Predicted Tg: {pred[0]:.2f} K")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            st.subheader("3D Polymer Structure")
            viewer = draw_3d_molecule(smiles)
            if viewer:
                viewer_html = viewer._make_html()
                st.components.v1.html(viewer_html, height=450, width=450)
            else:
                st.info("3D structure not available for this molecule.")

if __name__ == "__main__":
    main()
