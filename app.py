import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
import joblib
import torch
import torch.nn as nn
import py3Dmol

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same SimpleNN used during training
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Define the StackedModel class used during training
class StackedModel:
    def __init__(self, model_lgb, model_nn, stack_model, input_dim):
        self.model_lgb = model_lgb
        self.model_nn = model_nn
        self.stack_model = stack_model
        self.input_dim = input_dim

    def predict(self, X_np):
        lgb_pred = self.model_lgb.predict(X_np)
        self.model_nn.eval()
        with torch.no_grad():
            nn_pred = self.model_nn(torch.tensor(X_np, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)
        stacked_input = np.vstack([lgb_pred, nn_pred]).T
        return self.stack_model.predict(stacked_input)

# Load the stacked model and scaler
model = joblib.load('stacked_polymer_model.pkl')   # includes model_lgb, model_nn, and stack_model
scaler = joblib.load('polymer_feature_scaler.pkl')

# Molecular fingerprint + MACCS
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

# Molecule viewer
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

# Main Streamlit UI
def main():
    st.title("Polymer Tg Prediction + 3D Viewer")

    st.markdown("""
    This app predicts the **glass transition temperature (Tg)** of a polymer based on its **SMILES** and optional molecular properties.

    Enter:
    - SMILES string (required)
    - FFV, Tc, Density, Rg, and Molecular Weight (optional)

    Then click **Predict Tg and Show 3D**.
    """)

    # Input fields
    smiles = st.text_input("Enter Polymer SMILES:")
    st.subheader("Optional Molecular Properties")
    ffv = st.number_input("FFV (Free Volume Fraction)", min_value=0.0, step=0.01, format="%.4f")
    tc = st.number_input("Tc (Critical Temperature in K)", min_value=0.0, step=1.0)
    density = st.number_input("Density (g/cm³)", min_value=0.0, step=0.01, format="%.4f")
    rg = st.number_input("Rg (Radius of Gyration)", min_value=0.0, step=0.1)
    mw = st.number_input("Molecular Weight", min_value=0.0, step=1.0)

    if st.button("Predict Tg and Show 3D"):
        if not smiles:
            st.error("Please enter a valid SMILES string.")
            return

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES structure.")
            return

        st.write(f"Input SMILES: `{smiles}`")

        with st.spinner("Processing and predicting..."):
            features = featurize_fp(smiles)
            if features is None:
                st.error("Fingerprint generation failed.")
                return

            # Combine with additional features
            extra_features = [ffv, tc, density, rg, mw]
            full_features = np.concatenate([features, extra_features])

            try:
                features_scaled = scaler.transform([full_features])
            except Exception as e:
                st.error(f"Feature scaling failed: {e}")
                return

            try:
                pred = model.predict(features_scaled)
                st.success(f"Predicted Tg: **{pred[0]:.2f} K**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            # 3D visualization
            st.subheader("3D Polymer Structure")
            viewer = draw_3d_molecule(smiles)
            if viewer:
                viewer_html = viewer._make_html()
                st.components.v1.html(viewer_html, height=450, width=450)
            else:
                st.info("3D structure not available for this molecule.")

if __name__ == "__main__":
    main()
