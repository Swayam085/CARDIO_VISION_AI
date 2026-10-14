# CardioVision AI

> **AI-Powered Cardiovascular Risk Visualization & Multi-Vessel Stenosis Prediction**  
> *Multimodal AI Hackathon 2026 — Cardiovascular Track*

![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue?logo=python)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%2F%20Uvicorn-009688?logo=fastapi)
![React 19](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite%208-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/Language-TypeScript%20%2F%20Python-3178C6?logo=typescript)
![Three.js](https://img.shields.io/badge/3D-Three.js%20%2F%20R3F-black?logo=three.js)
![Tests](https://img.shields.io/badge/Pytest-324%20Passed-brightgreen?logo=pytest)
![Anti-Leakage](https://img.shields.io/badge/Anti--Leakage-100%25%20Guarded-red)
![License](https://img.shields.io/badge/License-MIT%20%2F%20CC%20BY%204.0-lightgrey)

---

## ⚠️ Clinical Safety Notice & Disclaimer

> [!CAUTION]
> **FOR EDUCATIONAL, RESEARCH, AND CLINICAL DECISION-SUPPORT DEMONSTRATION ONLY.**
>
> 1. This application generates **statistical model-based predictions** from clinical variables and does **NOT** provide a formal medical diagnosis or treatment directive.
> 2. It does **NOT** replace certified clinical evaluation, catheter angiography, diagnostic imaging, or the clinical judgment of licensed healthcare providers.
> 3. The 3D anatomical visualization displays **vessel-level statistical risk probabilities** mapped to canonical reference nodes (BodyParts3D FMA ontology). It does **NOT** represent patient-specific CT/MRI voxel segmentation, intravascular ultrasound (IVUS), or exact spatial plaque coordinates.

---

## 📌 Executive Summary

**CardioVision AI** is an end-to-end clinical machine learning and 3D visualization platform designed to estimate coronary artery disease (CAD) and vessel-specific stenosis risk (**LAD**, **LCX**, **RCA**) using **routine, non-invasive clinical features exclusively**.

Catheter angiography is an invasive, resource-intensive procedure carrying procedural risks. CardioVision AI stratifies multi-vessel risk early in the patient pathway using 55 standard clinical biomarkers—incorporating demographics, symptom presentation, 12-lead ECG markers, serum biochemistry, and echocardiography—while delivering real-time **TreeSHAP explainability** and an interactive **3D coronary digital twin**.

---

## 🏆 Key Features

- 🫀 **Multi-Target Coronary Risk Estimation**: Simultaneously predicts global CAD alongside vessel-specific stenosis probabilities for the three primary coronary arteries (**LAD**, **LCX**, **RCA**).
- 🛡️ **Zero Data-Leakage Architecture**: Catheterization outcomes (`LAD`, `LCX`, `RCA`, `Cath`) are strictly excluded from input spaces with runtime mathematical guards (`HTTP 422 DataLeakageError`).
- 📊 **Stratified 5-Fold Cross-Validation**: Validated across all 303 patients from the official UCI Dataset ID 411 with fold-isolated preprocessing, threshold tuning on out-of-fold (OOF) predictions, and zero target contamination.
- 🔍 **Real-Time Explainable AI (SHAP)**: High-speed TreeSHAP attribution engine delivering patient-level waterfall plots, positive risk exacerbators, protective factors, and feature contribution rankings.
- 🌐 **Interactive 3D Heart Digital Twin**: Built with Three.js and React Three Fiber, featuring authentic BodyParts3D FMA anatomical mapping, animated shader risk heatmaps, orbital controls, and vessel isolation.
- 📋 **Clinical Consultation Report**: Comprehensive audit trail summarizing patient biomarkers, vessel risk profiles, SHAP feature drivers, and clinical review signoff with one-click print/PDF export.
- 🧪 **Enterprise Test Suite**: 324 automated Pytest tests validating schema constraints, adversarial inputs, zero-leakage policies, and model calibration, coupled with strict TypeScript validation.

---

## 🔬 Official Data Provenance

| Parameter | Specification |
|---|---|
| **Dataset Name** | Extension of Z-Alizadeh Sani CAD Dataset |
| **Repository** | UCI Machine Learning Repository (Dataset ID: 411) |
| **Canonical URL** | [https://archive.ics.uci.edu/dataset/411/extention+of+z](https://archive.ics.uci.edu/dataset/411/extention+of+z) |
| **Cohort Size** | 303 Patients |
| **Input Feature Count** | 55 Non-Leaked Clinical Features |
| **Target Variables** | 4 Binary Targets: `Cath` (CAD), `LAD`, `LCX`, `RCA` |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Citation** | Z-Alizadeh Sani, M., et al. (2022). Extension of Z-Alizadeh Sani Dataset. *UCI Machine Learning Repository*. |

### 55 Non-Leaked Clinical Input Features

1. **Demographics (4)**: `Age`, `Sex`, `Weight`, `Length`, `BMI`
2. **Medical History & Comorbidities (13)**: `DM`, `HTN`, `Current Smoker`, `EX-Smoker`, `FH`, `Obesity`, `CRF`, `CVA`, `Airway disease`, `Thyroid Disease`, `CHF`, `DLP`, `BP`
3. **Physical & Vital Signs (7)**: `PR`, `Edema`, `Weak Peripheral Pulse`, `Lung rales`, `Systolic Murmur`, `Diastolic Murmur`, `Function Class`
4. **Symptom Presentation (5)**: `Typical Chest Pain`, `Dyspnea`, `Atypical`, `Nonanginal`, `Exertional CP`, `LowTH Ang`
5. **ECG & Rhythm Markers (7)**: `Q Wave`, `St Elevation`, `St Depression`, `Tinversion`, `LVH`, `Poor R Progression`, `BBB`
6. **Laboratory Biochemistry (14)**: `FBS`, `CR`, `TG`, `LDL`, `HDL`, `BUN`, `ESR`, `HB`, `K`, `Na`, `WBC`, `Lymph`, `Neut`, `PLT`
7. **Echocardiography (3)**: `VHD`, `EF-TTE`, `Region RWMA`

---

## 📈 Machine Learning Benchmarks & Model Selection

All models were evaluated via **Stratified 5-Fold Cross-Validation** on UCI Dataset 411 ($N=303$). Preprocessing pipelines (`SimpleImputer`, `StandardScaler`, `OrdinalEncoder`) were fitted strictly inside training folds to prevent data leakage. Thresholds were optimized on Out-of-Fold (OOF) predictions using a composite metric ($0.70 \times \text{ROC-AUC} + 0.30 \times \text{F1}$).

### Verified Out-of-Fold (OOF) Performance

| Target | Clinical Meaning | Selected Model | Optimal Cutoff | ROC-AUC | F1-Score | Accuracy | Recall | Precision | Avg Precision (PR-AUC) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CAD** | Overall Coronary Artery Disease | **Random Forest** | `0.61` | **0.9212** | **0.9052** | **86.80%** | **88.43%** | **92.72%** | **0.9657** |
| **LAD** | Left Anterior Descending Stenosis | **Random Forest** | `0.51` | **0.8436** | **0.8387** | **80.20%** | **88.14%** | **80.00%** | **0.8758** |
| **LCX** | Left Circumflex Stenosis | **XGBoost** | `0.19` | **0.7458** | **0.6450** | **64.03%** | **83.19%** | **52.66%** | **0.6233** |
| **RCA** | Right Coronary Artery Stenosis | **Logistic Regression** | `0.37` | **0.7003** | **0.6071** | **63.70%** | **74.56%** | **51.20%** | **0.5596** |

> **Threshold Strategy Note:** Default cutoffs (`0.50`) yield high recall but suboptimal balance for imbalanced vessel lesions. Threshold optimization on OOF predictions recalibrated decision boundaries (e.g., LCX at `0.19` and RCA at `0.37`), significantly boosting clinical sensitivity for occult lesions.

---

## 🛡️ Anti-Leakage Protocol

In coronary artery disease research, inadvertent inclusion of catheterization results in model features invalidates clinical utility. CardioVision AI implements a multi-layer firewall:

```
[ Ingested Data / API Payload ]
               │
               ▼
   [ Anti-Leakage Firewall ]
   ┌───────────────────────────────────────────────┐
   │ Banned Columns:                               │
   │  - LAD   (Left Anterior Descending)           │
   │  - LCX   (Left Circumflex)                    │
   │  - RCA   (Right Coronary Artery)              │
   │  - Cath  (Coronary Angiography Summary)       │
   └───────────────────────────────────────────────┘
               │
               ├─► [Contains Banned Token?] ──► 🚨 HTTP 422 DataLeakageError (Rejected)
               │
               ▼
   [ Validated 55 Clinical Features Only ]
```

1. **Data Ingestion**: Excluded prior to exploratory analysis or model training.
2. **Pydantic Validation**: Payload schemas reject any payload containing banned keys.
3. **Automated Audit**: Automated tests check zero feature overlap between input vectors and targets.

---

## 🏗️ System Architecture & Workflow

```
[ Clinical Input Vector (55 Features) ]
 (Demographics, Medical History, Vitals, Symptoms, ECG, Labs, Echo)
               │
               ▼
[ FastAPI Backend Engine — /api/v1/analyze ]
 ├── 1. Schema Validation (Pydantic v2 Range & Type Guards)
 ├── 2. Anti-Leakage Firewall (Zero Banned Tokens)
 └── 3. Preprocessing Pipeline (preprocessor.joblib: Imputation, One-Hot Encoding, Scaling)
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
[ 4 Trained ML Models ]          [ TreeSHAP Engine ]
 ├── CAD (Random Forest)          ├── Fast TreeExplainer
 ├── LAD (Random Forest)          ├── SamplingExplainer Fallback
 ├── LCX (XGBoost)                └── Attributions & Base Values
 └── RCA (Logistic Regression)                  │
       │                                        │
       └─────────────────┬──────────────────────┘
                         ▼
[ Unified JSON Response Payload ]
 (Predictions, Probabilities, Threshold Cutoffs, Risk Tiers, SHAP Values)
                         │
                         ▼
[ Frontend Application — React 19 + Vite 8 SPA ]
 ├── 🎛️ Analysis Hub (/analyze): 55-feature categorized clinical form & pre-filled presets
 ├── 📊 Scorecards (/results): Multi-vessel risk gauges, cutoffs, and summary recommendations
 ├── 🫀 3D Canvas (/visualization): Three.js interactive heart with BodyParts3D FMA mapping
 ├── 🔬 Explainability (/insights): Waterfall attribution charts & radar factor comparisons
 ├── 📈 Model Metrics (/metrics): Interactive ROC curves, PR curves, and 5-fold CV spreads
 └── 📑 Clinical Audit (/report): Physician consultation report with PDF / print export
```

---

## 🫀 3D Anatomical Heart & Ontology Mapping

The 3D visualization is built with **Three.js** and **React Three Fiber**, referencing anatomical models from **BodyParts3D / Anatomography** (Database Center for Life Science, Japan).

| Vessel | Anatomical Name | Foundational Model of Anatomy (FMA) ID | Dynamic Color Gradient |
|:---|:---|:---:|:---|
| **LAD** | Left Anterior Descending Branch | `FMA74912` | Green (`<0.30`) → Amber (`0.30–0.50`) → Red (`>0.51`) |
| **LCX** | Circumflex Branch of Left Coronary Artery | `FMA74923` | Green (`<0.15`) → Amber (`0.15–0.18`) → Red (`>0.19`) |
| **RCA** | Right Coronary Artery | `FMA3802` | Green (`<0.25`) → Amber (`0.25–0.36`) → Red (`>0.37`) |
| **Myocardium / Aorta** | Base Anatomical Structure | `FMA7088` / `FMA3734` | Semi-translucent anatomical neutral |

- **Shader Risk Heatmap**: Real-time shader interpolation maps model-predicted probabilities to vessel surface meshes.
- **Interactive Orbit Controls**: 360-degree rotation, zoom, pan, wireframe mode, and individual vessel focus toggles.
- **Anatomy Inspector**: Displays vessel length, clinical perfusion territory, and stenosis probability.

---

## 💻 Tech Stack Overview

### Frontend
- **Framework**: React 19, TypeScript
- **Bundler & Tooling**: Vite 8, Oxlint
- **3D Graphics**: Three.js, `@react-three/fiber`, `@react-three/drei`
- **Data Visualization**: Recharts
- **Styling**: Tailwind CSS v3 (custom clinical dark-mode theme)
- **Routing**: React Router DOM v7 (code-split lazy loading)
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI (Asynchronous Python 3.11 / 3.14)
- **Validation**: Pydantic v2
- **Server**: Uvicorn (ASGI)
- **ML Frameworks**: Scikit-Learn 1.8+, XGBoost 3.3+, LightGBM 4.6+, Joblib
- **Explainability**: SHAP (TreeExplainer, SamplingExplainer)

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Python**: 3.11 or higher
- **Node.js**: 20 or higher
- **Git**

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/swayamchoksi/CARDIOVISION_AI.git
cd CARDIOVISION_AI
```

---

### Step 2: Backend Setup & Execution

1. Open a terminal and navigate to `backend/`:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment:
   ```bash
   # Verify .env exists in project root (or copy from .env.example)
   cp ../.env.example ../.env
   ```

5. Launch the FastAPI backend server:
   ```bash
   python main.py
   ```
   - API Root: `http://localhost:8000`
   - Interactive Swagger Docs: `http://localhost:8000/docs`
   - Service Health Check: `http://localhost:8000/health`

---

### Step 3: Frontend Setup & Execution

In a second terminal window:

1. Navigate to `frontend/`:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Launch the Vite development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5173
   ```

---

## 🧪 Comprehensive Verification & QA

### 1. Backend & ML Pytest Suite (324 Tests)
Execute the complete test suite from the root directory:
```bash
# In workspace root with virtual environment activated:
pytest tests/ -v
```
**Verification Coverage:**
- **Zero-Leakage Assurance**: Mathematically enforces complete exclusion of catheterization outcomes from input vectors.
- **Adversarial & Boundary Testing**: Rejects invalid physiological values (e.g., negative BP, extreme heart rates).
- **Probability Bounds**: Verifies all model outputs strictly adhere to $[0.0, 1.0]$.
- **SHAP Additivity**: Confirms $\sum \text{SHAP} + \text{Base Value} \approx f(x)$.

### 2. Frontend Type-Check & Production Build
```bash
cd frontend
npm run type-check
npm run build
```
- Performs strict TypeScript verification (`tsc --noEmit`).
- Bundles and tree-shakes client assets into `frontend/dist/`.

---

## 📖 API Documentation Reference

All endpoints are registered under both root and `/api/v1/`:

| Endpoint | Method | Input | Output | Description |
|---|:---:|---|---|---|
| `/health` | `GET` | *None* | `HealthResponse` | Liveness check, uptime, and model loading status |
| `/api/v1/analyze` | `POST` | `PatientData` (55 features) | `AnalyzeResponse` | **Unified Pipeline**: Predictions, cutoffs, and SHAP attributions in a single call |
| `/api/v1/predict` | `POST` | `PatientData` (55 features) | `PredictionResponse` | Returns multi-vessel probabilities, binary decisions, and risk levels |
| `/api/v1/explain` | `POST` | `ExplainRequest` | `ExplainResponse` | TreeSHAP attributions for specified targets |
| `/api/v1/metrics` | `GET` | *None* | `MetricsResponse` | 5-fold CV metrics, confusion matrices, and ROC/PR data |
| `/api/v1/model-info` | `GET` | *None* | `ModelInfoResponse` | Model versions, hyper-parameters, and feature lists |

---

## 📂 Repository Structure

```
CARDIOVISION_AI/
├── frontend/                        # React 19 + TypeScript + Vite 8 SPA
│   ├── public/models/               # BodyParts3D GLB anatomical heart model
│   ├── src/
│   │   ├── components/              # UI components, 3D Canvas, and charts
│   │   ├── context/                 # AnalysisContext state management
│   │   ├── pages/                   # Route pages (Home, Analyze, Results, 3D, Insights, Metrics, Report)
│   │   ├── services/                # Axios API client
│   │   └── types/                   # TypeScript interfaces and schemas
│   ├── package.json                 # Node dependencies and scripts
│   └── vite.config.ts               # Vite configuration and build optimizations
│
├── backend/                         # FastAPI Application
│   ├── app/
│   │   ├── api/                     # REST API route handlers
│   │   ├── schemas/                 # Pydantic request/response schemas
│   │   ├── services/                # Inference service & SHAP explainability service
│   │   └── utils/                   # Configuration and logging
│   ├── artifacts/                   # Serialized models (.joblib), metadata, and CV metrics
│   ├── main.py                      # FastAPI server entry point
│   └── requirements.txt             # Backend dependencies
│
├── ml/                              # Machine Learning Pipeline
│   ├── data/raw/                    # Raw UCI Dataset 411 files
│   ├── src/
│   │   ├── download_data.py         # Automated UCI ID 411 downloader
│   │   ├── feature_config.py        # 55-feature schema & anti-leakage guards
│   │   ├── leakage_audit.py         # Static code analysis for data leakage
│   │   └── train.py                 # Stratified 5-fold CV training pipeline
│   └── artifacts/                   # Training exports and evaluation artifacts
│
├── tests/                           # Complete Test Suite (324 Tests)
│   ├── backend/                     # API validation, endpoints, and error handling tests
│   ├── ml/                          # Zero-leakage, CV bounds, and pipeline tests
│   └── test_end_to_end_pipeline.py  # Integrated pipeline tests
│
├── docs/                            # In-Depth Documentation
│   ├── 3d-assets.md                 # BodyParts3D FMA anatomy mapping guide
│   ├── DATA_SOURCES.md              # UCI Dataset 411 clinical descriptions
│   └── QA_REPORT.md                 # Full test and QA validation report
│
├── .env.example                     # Environment template
├── vercel.json                      # Vercel deployment configuration
└── README.md                        # Master Project Documentation
```

---

## 📜 Academic Citations & Acknowledgments

```bibtex
@misc{alizadehsani2022extension,
  author       = {Z-Alizadeh Sani, Zahra and others},
  title        = {Extension of Z-Alizadeh Sani Dataset},
  year         = {2022},
  publisher    = {UCI Machine Learning Repository},
  doi          = {10.24432/C5QS3G},
  url          = {https://archive.ics.uci.edu/dataset/411/extention+of+z}
}

@article{lundberg2020local2global,
  title        = {From local explanations to global understanding with explainable AI for trees},
  author       = {Lundberg, Scott M and Erion, Gabriel and Chen, Hugh and others},
  journal      = {Nature Machine Intelligence},
  volume       = {2},
  number       = {1},
  pages        = {56--67},
  year         = {2020},
  publisher    = {Nature Publishing Group}
}
```

- **3D Anatomical Heart**: BodyParts3D, © Database Center for Life Science (DBCLS) licensed under `CC BY-SA 2.1 JP`.
- **Foundational Model of Anatomy (FMA)**: Structural Informatics Group, University of Washington.

---

## 📄 Licensing

- **Software Code**: [MIT License](https://opensource.org/licenses/MIT)
- **Clinical Dataset**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **3D Anatomical Assets**: [Creative Commons Attribution-ShareAlike 2.1 Japan (CC BY-SA 2.1 JP)](https://creativecommons.org/licenses/by-sa/2.1/jp/)
