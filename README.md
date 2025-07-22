# Polymer Tg Predictor + 3D Viewer 🌡️🧬

A Streamlit web app that predicts the **glass transition temperature (Tg)** of polymers using their **SMILES notation** and additional molecular properties. The app also provides a **3D visualization** of the polymer's molecular structure.

---

## 🔬 What is Tg?

**Tg (Glass Transition Temperature)** is the temperature at which a polymer transitions from a hard, glassy state to a soft, rubbery state. It’s a crucial property in polymer design, affecting flexibility, durability, and thermal stability.

---

## 🧠 Features

- 🔮 **Tg Prediction** using a trained XGBoost machine learning model
- 🧪 **Input Polymer SMILES** and additional molecular descriptors:
  - Free Volume Fraction (FFV)
  - Critical Temperature (Tc)
  - Density
  - Radius of Gyration (Rg)
  - Molecular Weight
- 🧬 **3D Molecular Visualization** using RDKit and py3Dmol
- 💡 Error handling for invalid inputs
- 🧮 Built-in StandardScaler and feature processing

---

## 🛠️ Tech Stack

- Python 3.10
- [Streamlit](https://streamlit.io/)
- [RDKit](https://www.rdkit.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [py3Dmol](https://3dmol.csb.pitt.edu/)
- NumPy

---

## 📦 Installation & Setup

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/polymer-tg-predictor.git
cd polymer-tg-predictor
```

### 2. (Optional) Create a virtual environment:

```bash
conda create -n polymer_env python=3.10
conda activate polymer_env
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** RDKit may require conda installation. Use:
```bash
conda install -c rdkit rdkit
```

---

## 🚀 Running the App Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ☁️ Deploying on Streamlit Cloud

1. Push your code to a **GitHub repository**
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Click **New App**
4. Connect your GitHub repo and deploy!

---

## 🧪 Sample Input

Try testing with this example:

**SMILES:** `CC(C)C(=O)O`  
**FFV:** `0.12`  
**Tc:** `320`  
**Density:** `1.05`  
**Rg:** `4.5`  
**Molecular Weight:** `120.0`

---

## 📁 Project Structure

```
📦 polymer-tg-predictor/
├── app.py                  # Main Streamlit app
├── polymer_xgb_model_optimized.pkl  # Trained XGBoost model
├── polymer_scaler.pkl      # Trained StandardScaler
├── requirements.txt
└── README.md
```

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it!

---

## 🙌 Acknowledgements

- [NeurIPS 2025 Polymer Prediction Challenge](https://www.kaggle.com/competitions/neurips-2025-open-polymer)
- [RDKit](https://www.rdkit.org/)
- [py3Dmol](https://3dmol.csb.pitt.edu/)
- [Streamlit](https://streamlit.io/)
