# ALLOY IQ 🧪🔬
> **AI-Powered Materials Property Prediction & Metallurgical Microstructure Generation**

[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2015-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Blender](https://img.shields.io/badge/Rendering-Blender%204.x-orange?style=flat-square&logo=blender)](https://www.blender.org/)
[![Python](https://img.shields.io/badge/ML%20Engine-Python%203.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**ALLOY IQ** is an advanced, integrated software suite designed for modern materials science and metallurgy. It combines automated multi-tiered data ingestion, machine learning-driven mechanical property prediction (with SHAP-based explanations), and procedural high-fidelity 3D microstructure rendering in Blender.

---

## 🌟 Key Features

### 1. 📊 Automated Metallurgical Ingestion Pipeline
- **Multi-Source Fetching**: Automatically collects high-entropy alloy (HEA) and crystalline data from OQMD, AFLOW, Kaggle, and curated GitHub MPEA repositories.
- **AI-Powered Literature Scraper**: Built-in PDF parser and literature scraping engine to automatically ingest and structure data directly from metallurgical research papers.
- **Robust Normalization**: Automatic iron-balance calculation, alloy stoichiometry validation, and outlier detection with systematic error logging.

### 2. 🧠 Machine Learning Backend (FastAPI)
- **High-Sparsity Imputation**: Built-in strategies designed specifically for sparse alloy composition matrices.
- **Ensemble Model Training**: Integrated training pipeline utilizing XGBoost, Random Forest, Multi-Layer Perceptrons, and hyperparameter optimization via Optuna.
- **Explainable AI (XAI)**: Immediate SHAP-based model explanations accompanying predictions to show the impact of individual element ratios on mechanical strength.
- **Production Endpoints**: Real-time property predictions (`/predict/mechanical`) and historical query tracing with a fully documented Swagger UI.

### 3. 🎨 Procedural Microstructure Generator (Blender)
- **Voronoi Tessellation**: Generates true crystallographic grains using SciPy-driven Voronoi mathematical algorithms.
- **Physics-Informed Carbide Placement**: Procedurally scatters strengthening carbides along grain boundary contours to match real metallurgical structures.
- **Advanced Shaders**: Procedural PBR material node groups mimicking etching relief depth and grain-specific crystallographic orientations.

### 4. 💻 Next.js Enterprise Dashboard
- **Modern Dark UI**: A glassmorphic, responsive user interface built using modern React principles.
- **Interactive Predictions**: Real-time forms to test alloy recipes and visualize property distributions instantly.
- **Historical Query Audit**: Complete interactive logger of previous compositions, predictions, and model performance.

---

## 📂 Repository Architecture

```text
ALLOY IQ/
├── backend/                  # FastAPI web server and ML core
│   ├── ingestion/            # AFLOW, OQMD, literature scraper, and data pipelines
│   ├── ml/                   # XGBoost, RandomForest, Optuna engines & SHAP explainers
│   ├── data/                 # Feature extraction & stoichiometric helper code
│   └── tests/                # Pipeline smoke tests and API validation
├── frontend/                 # Next.js 15 web dashboard
│   ├── src/app/              # Next.js App Router (Predict, History, Microstructure pages)
│   ├── src/components/       # Reusable UI components (Navbar, widgets)
│   └── package.json          # Node dependencies
├── blender/                  # Procedural metallurgical rendering scripts
│   ├── microstructure_generator.py     # 3D Voronoi grain rendering
│   └── implementation_plan.md          # Shader & geometry pipelines
├── models/                   # Local serialized model weights (Git ignored)
└── docker-compose.yml        # Orchestration configuration
```

---

## ⚙️ Quick Start Guide

### Prerequisites
- **Python**: `3.10` or `3.11`
- **Node.js**: `18.x` or `20.x`
- **Blender**: `4.x` (with `scipy` and `numpy` installed in its Python environment)

### 1. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the API server:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8005 --reload
   ```
   Open [http://localhost:8005/docs](http://localhost:8005/docs) to access the interactive Swagger API documentation.

### 2. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   Open [http://localhost:3001](http://localhost:3001) to interact with the modern dashboard!

### 3. Blender Microstructure Generation
To run the procedural rendering pipeline:
1. Open Blender in headless mode or use the script within the Blender Python Console:
   ```bash
   blender --background --python blender/microstructure_generator.py
   ```

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created by [TheShriHari](https://github.com/TheShriHari)*
