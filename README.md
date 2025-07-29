# 🧪 Polymer Tg Prediction App

This Streamlit application predicts the **glass transition temperature (Tg)** of polymers from their SMILES (Simplified Molecular Input Line Entry System) representations. Built using machine learning techniques, the app combines cheminformatics and regression modeling to offer rapid Tg estimation with 3D molecular visualization.

---

## 📌 Key Features

- 🔬 **Tg Prediction** from SMILES input
- 🧬 Uses **Morgan Fingerprints**, **MACCS Keys**, and RDKit **descriptors**
- 🤖 Powered by a **stacked ML model** (LightGBM, PyTorch NN, Linear Regression)
- 💠 Interactive **3D molecular structure viewer** using `py3Dmol`
- 🖥️ Streamlit-based **web interface**

---

## 📁 Repository Structure

```
polymer-tg-app/
│
├── app.py                           # Streamlit frontend
├── utils/
│   └── featurize.py                 # RDKit-based SMILES featurization
├── models/
│   ├── full_stacked_model.pkl      # Trained ML model
│   └── feature_scaler.pkl          # Feature scaler
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### ✅ Using Anaconda (Recommended)

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/polymer-tg-app.git
   cd polymer-tg-app
   ```

2. **Create and activate the environment**
   ```bash
   conda create -n polymer_tg_env python=3.10
   conda activate polymer_tg_env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   conda install -c rdkit rdkit
   ```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Overview

- **Input**: SMILES strings (e.g., `CCO`, `c1ccccc1C(C)C`)
- **Features**:
  - Morgan fingerprints (2048 bits)
  - MACCS keys (167 bits)
  - 12 molecular descriptors (e.g., MolWt, NumHDonors)
- **Model**:
  - LightGBM
  - PyTorch Feedforward Neural Network
  - Linear Regression stacking
- **Target**: Tg (Glass Transition Temperature, Kelvin)

---

## 🔬 Example Inputs & Outputs

| SMILES             | Predicted Tg (K) |
|--------------------|------------------|
| `CCO`              | 42.68            |
| `C(CO)OCCO`         | 15.14            |
| `c1ccccc1C(C)C`     | 107.66           |
| `CC(C)CC(=O)O`      | ~131.10          |

---

## 📊 Sample Screenshot

![App Screenshot](https://your-placeholder-screenshot-url.com/screenshot.png)

---

## 📚 References

- Afzal, M. A. F. et al. (2019) ‘A polymer informatics approach to discovering novel dielectric polymers’, *Computational Materials Science*, 160, pp. 329–336.
- Chen, L., Pilania, G., and Ramprasad, R. (2020) ‘Polymer Genome: A data-powered polymer informatics platform for property predictions’, *Computational Materials Science*, 158, pp. 30–38.
- Ramprasad, R. et al. (2017) ‘Machine learning in materials informatics: recent applications and prospects’, *npj Computational Materials*, 3(1), pp. 1–13.

---

## 🧑‍💻 Author

Developed by **Pramod Sandakelum**  
🔗 Contact: [your.email@example.com](mailto:your.email@example.com)

---

## 📃 License

MIT License. See [LICENSE](LICENSE) for details.
