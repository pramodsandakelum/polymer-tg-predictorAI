import streamlit as st
import joblib
import numpy as np
import torch
import torch.nn as nn
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

# Streamlit app UI
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Container for flex layout */
    .container-flex {
        display: flex;
        justify-content: center;
        gap: 3rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    /* Left pane: 3D viewer */
    .viewer-container {
        border: 2px solid #3498db;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 0 15px rgba(52, 152, 219, 0.2);
        max-width: 450px;
        height: 350px;
        flex-shrink: 0;
    }
    /* Right pane: prediction cards grid */
    .pred-grid {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        max-width: 300px;
    }
    .pred-card {
        background: #ecf0f1;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: transform 0.2s ease;
    }
    .pred-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }
    .pred-title {
        font-weight: 700;
        color: #2980b9;
        margin-bottom: 0.25rem;
        font-size: 1.2rem;
    }
    .pred-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #27ae60;
    }
    /* Footer */
    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.9rem;
        color: #95a5a6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<h1 class="title">Polymer Property Predictor with 3D Viewer 🔬</h1>', unsafe_allow_html=True)

# Input box with help tooltip
smiles_input = st.text_input(
    label="Enter a polymer SMILES string:",
    placeholder="e.g. C(C(=O)O)N (Glycine)",
    key="smiles_input",
    help="Input the SMILES notation of your polymer molecule. Example: C(C(=O)O)N"
)

def show_3d_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES: Could not parse molecule.")
        return
    mb = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=300)
    viewer.addModel(mb, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    html = viewer._make_html()
    return html

if smiles_input:
    html_3d = show_3d_molecule(smiles_input)
    # Featurize & predict
    fps = featurize_combo(smiles_input)
    desc = calc_extended_descriptors(smiles_input)
    features = np.hstack([fps, desc]).reshape(1, -1)
    preds = full_model.predict(features)

    # Render side-by-side flexbox layout
    st.markdown('<div class="container-flex">', unsafe_allow_html=True)

    # Left: 3D viewer container
    st.markdown('<div class="viewer-container">', unsafe_allow_html=True)
    st.components.v1.html(html_3d, height=350)
    st.markdown('</div>', unsafe_allow_html=True)

    # Right: Predictions cards
    st.markdown('<div class="pred-grid">', unsafe_allow_html=True)
    for i, target in enumerate(targets):
        st.markdown(
            f'''
            <div class="pred-card">
                <div class="pred-title">{target}</div>
                <div class="pred-value">{preds[0, i]:.4f}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close container-flex

# Footer
st.markdown(
    '<div class="footer">Made with ❤️ by Pramod • Powered by Streamlit and RDKit</div>',
    unsafe_allow_html=True,
)
