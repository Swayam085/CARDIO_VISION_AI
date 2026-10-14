/**
 * CardioVision AI — Technical & Clinical Documentation (/documentation)
 * ======================================================================
 * Complete documentation of the CardioVision AI system covering:
 *   1. Problem Formulation & Clinical Context
 *   2. Dataset Provenance & Cohort Demographics (UCI ID 411)
 *   3. Preprocessing Pipeline & Feature Encodings
 *   4. Data Leakage Prevention Protocol
 *   5. Machine Learning Models & Ensemble Architectures
 *   6. Validation Methodology & Out-of-Fold Tuning
 *   7. Interpretability via TreeSHAP
 *   8. FastAPI Backend Architecture & API Specifications
 *   9. Frontend Architecture & Design System
 *  10. 3D Anatomical Visualization Engine (Three.js / BodyParts3D)
 *  11. Vessel-to-Model Risk Mapping Protocol
 *  12. Critical System Limitations
 *  13. Explicit 3D Anatomical Limitations
 *  14. Safety Governance & Decision-Support Classification
 *  15. Reproducibility & Pipeline Determinism
 */

import { useState } from 'react';
import {
  Database,
  Shield,
  Activity,
  Cpu,
  Layers,
  Heart,
  FileCode2,
  AlertTriangle,
  CheckCircle2,
  Lock,
  Boxes,
  Compass,
  Zap,
  ShieldAlert,
  ExternalLink,
  BookOpen,
  GitBranch,
} from 'lucide-react';

const SECTIONS = [
  { id: 'problem',           label: '1. Problem' },
  { id: 'dataset',           label: '2. Dataset' },
  { id: 'preprocessing',     label: '3. Preprocessing' },
  { id: 'leakage',           label: '4. Leakage Prevention' },
  { id: 'models',            label: '5. ML Models' },
  { id: 'validation',        label: '6. Validation' },
  { id: 'shap',              label: '7. SHAP' },
  { id: 'fastapi',           label: '8. FastAPI Architecture' },
  { id: 'frontend',          label: '9. Frontend' },
  { id: 'visualization-3d',  label: '10. 3D Visualization' },
  { id: 'vessel-mapping',    label: '11. Vessel Mapping' },
  { id: 'limitations',       label: '12. Limitations' },
  { id: 'limitations-3d',    label: '13. 3D Limitation' },
  { id: 'safety',            label: '14. Safety' },
  { id: 'reproducibility',   label: '15. Reproducibility' },
];

export function DocumentationPage() {
  const [activeSection, setActiveSection] = useState<string>('problem');

  const scrollToSection = (id: string) => {
    setActiveSection(id);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <main id="documentation-page" className="section-container py-12 animate-fade-in space-y-10">
      {/* ── Document Header ─────────────────────────────────── */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-500/15 border border-primary-500/30 text-primary-400 text-xs font-mono font-medium">
          <BookOpen className="w-3.5 h-3.5" />
          <span>System Specifications & Architecture</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-display font-bold text-white tracking-tight">
          CardioVision AI Documentation
        </h1>
        <p className="text-gray-400 text-sm leading-relaxed max-w-2xl mx-auto">
          Comprehensive technical, algorithmic, and clinical architecture documentation for
          multi-vessel coronary risk estimation, TreeSHAP interpretability, and 3D anatomical projection.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* ── Sticky Table of Contents Sidebar ────────────────── */}
        <aside className="lg:col-span-1 hidden lg:block">
          <div className="sticky top-24 glass-card p-4 space-y-2 border-surface-border">
            <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-gray-400 block px-2 pb-1 border-b border-surface-border">
              Table of Contents
            </span>
            <nav className="space-y-1 text-xs font-mono">
              {SECTIONS.map((sec) => (
                <button
                  key={sec.id}
                  onClick={() => scrollToSection(sec.id)}
                  className={`w-full text-left px-2.5 py-1.5 rounded-lg transition-colors truncate block ${
                    activeSection === sec.id
                      ? 'bg-primary-500/20 text-primary-300 font-bold border border-primary-500/40'
                      : 'text-gray-400 hover:text-white hover:bg-surface-dark/60'
                  }`}
                >
                  {sec.label}
                </button>
              ))}
            </nav>
          </div>
        </aside>

        {/* ── Main Content Area ───────────────────────────────── */}
        <div className="lg:col-span-3 space-y-10 text-gray-300 text-sm leading-relaxed">

          {/* 1. PROBLEM */}
          <section id="problem" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-primary-400">
              <Heart className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">1. Clinical Problem & Motivation</h2>
            </div>
            <p>
              Coronary Artery Disease (CAD) remains the single leading cause of mortality worldwide. CAD occurs when
              atherosclerotic plaques accumulate inside coronary arteries, restricting oxygenated myocardial perfusion and
              predisposing patients to angina, myocardial infarction, and heart failure.
            </p>
            <p>
              In current clinical practice, definitive CAD diagnosis requires invasive coronary catheterization (Cath), which
              carries procedural risks, radiation exposure, and substantial healthcare costs. Non-invasive pre-test probability
              models often rely exclusively on binary overall risk scores, failing to provide artery-specific stenosis risk
              for the three primary coronary branches:
            </p>
            <ul className="list-disc pl-5 space-y-1 text-xs font-mono text-gray-300">
              <li><strong>LAD (Left Anterior Descending):</strong> Supplies 45–55% of left ventricular myocardium including anterior wall and septum.</li>
              <li><strong>LCX (Left Circumflex):</strong> Supplies the posterolateral left ventricle and anterolateral papillary muscle.</li>
              <li><strong>RCA (Right Coronary Artery):</strong> Supplies the right ventricle, inferior left ventricular wall, and cardiac conduction nodes.</li>
            </ul>
            <p>
              <strong>CardioVision AI</strong> bridges this gap by integrating 55 routine non-invasive clinical attributes
              (demographic, symptomatologic, laboratory, electrocardiographic, and echocardiographic) to estimate both global
              CAD presence and vessel-level stenosis risk across LAD, LCX, and RCA.
            </p>
          </section>

          {/* 2. DATASET */}
          <section id="dataset" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-emerald-400">
              <Database className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">2. Dataset Provenance & Cohort Characteristics</h2>
            </div>
            <p>
              The models are trained strictly on the <strong>Extension of Z-Alizadeh Sani Dataset</strong> hosted on the
              UCI Machine Learning Repository (Dataset ID: 411).
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono bg-surface-darker p-4 rounded-xl border border-surface-border">
              <div>
                <span className="text-gray-500 block">Dataset Citation:</span>
                <span className="text-gray-200">Alizadehsani et al. (2013), Tehran Heart Center cohort.</span>
              </div>
              <div>
                <span className="text-gray-500 block">Repository Link:</span>
                <a
                  href="https://archive.ics.uci.edu/dataset/411/extention+of+z"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary-400 hover:underline flex items-center gap-1"
                >
                  <span>UCI Repository ID 411</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
              <div>
                <span className="text-gray-500 block">Total Patients:</span>
                <span className="text-white font-bold">303 patients (all with invasive angiography)</span>
              </div>
              <div>
                <span className="text-gray-500 block">Raw Columns:</span>
                <span className="text-white font-bold">59 attributes (55 predictors + 4 targets)</span>
              </div>
            </div>
            <p>
              <strong>Primary CAD Target Discovery:</strong> Exploratory Data Analysis verified that no column named <code className="text-primary-300 font-mono">CAD</code> exists
              in the raw dataset. The primary CAD ground truth is the <code className="text-primary-300 font-mono">Cath</code> column (values: "CAD" [216 patients] vs "Normal" [87 patients]).
              Individual vessel targets are <code className="text-rose-300 font-mono">LAD</code> (177 stenotic vs 126 normal), <code className="text-orange-300 font-mono">LCX</code> (119 stenotic vs 184 normal),
              and <code className="text-blue-300 font-mono">RCA</code> (114 stenotic vs 189 normal).
            </p>
          </section>

          {/* 3. PREPROCESSING */}
          <section id="preprocessing" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-cyan-400">
              <Layers className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">3. Preprocessing Pipeline & Feature Encodings</h2>
            </div>
            <p>
              The 55 predictor features are divided into numerical and categorical streams and transformed through a reproducible
              scikit-learn <code className="text-cyan-300 font-mono">ColumnTransformer</code> pipeline:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-4 rounded-xl bg-surface-darker border border-surface-border space-y-2">
                <span className="text-cyan-400 font-bold block">Numerical Stream (34 Features)</span>
                <p className="text-gray-300 font-sans">
                  Age, Weight, Length, BMI, BP, PR, FBS, CR, TG, LDL, HDL, BUN, ESR, HB, K, Na, WBC, Lymph, Neut, PLT, EF-TTE, Region RWMA, etc.
                </p>
                <div className="p-2 rounded bg-surface-dark border border-surface-border/60 text-[11px] text-gray-400">
                  Pipeline: SimpleImputer(strategy='median') → StandardScaler()
                </div>
              </div>

              <div className="p-4 rounded-xl bg-surface-darker border border-surface-border space-y-2">
                <span className="text-purple-400 font-bold block">Categorical Stream (21 Features)</span>
                <p className="text-gray-300 font-sans">
                  Sex, Obesity, CRF, CVA, Airway disease, Thyroid Disease, CHF, DLP, Weak Peripheral Pulse, Lung rales, Systolic Murmur, Dyspnea, LVH, BBB, VHD, etc.
                </p>
                <div className="p-2 rounded bg-surface-dark border border-surface-border/60 text-[11px] text-gray-400">
                  Pipeline: SimpleImputer(strategy='most_frequent') → OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                </div>
              </div>
            </div>
            <p className="text-xs text-gray-400">
              All fitted transformers are persisted in <code className="text-gray-200 font-mono">backend/artifacts/preprocessor.joblib</code> ensuring
              deterministic, byte-identical input representations during online inference.
            </p>
          </section>

          {/* 4. LEAKAGE PREVENTION */}
          <section id="leakage" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-emerald-400">
              <Lock className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">4. Data Leakage Prevention Protocol</h2>
            </div>
            <p>
              In clinical machine learning, data leakage is a pervasive flaw where target proxies or downstream diagnostic outcomes
              are inadvertently included in predictor variables, inflating retrospective validation metrics:
            </p>
            <div className="p-4 rounded-xl bg-surface-darker border border-emerald-500/30 space-y-2 text-xs font-mono">
              <span className="text-emerald-400 font-bold block">UCI Documentation Note 1 Compliance:</span>
              <p className="text-gray-300 font-sans">
                The official UCI dataset documentation explicitly instructs: <em>"Only one of Cath, LAD, LCX, or RCA should be used as target; all others must be excluded from predictor inputs."</em>
              </p>
              <div className="flex flex-wrap gap-2 pt-1">
                {['Cath', 'LAD', 'LCX', 'RCA'].map((col) => (
                  <span key={col} className="px-2 py-0.5 rounded bg-red-500/20 text-red-300 border border-red-500/40 text-[10px]">
                    Excluded Predictor: {col}
                  </span>
                ))}
              </div>
            </div>
            <p>
              <strong>Cross-Validation Leakage Prevention:</strong> SMOTE (Synthetic Minority Over-sampling Technique)
              was <em>never applied before cross-validation splits</em>. Over-sampling was strictly performed exclusively on
              training folds within an <code className="text-emerald-300 font-mono">imblearn.pipeline.Pipeline</code>, leaving test folds pristine and uncorrupted.
            </p>
          </section>

          {/* 5. ML MODELS */}
          <section id="models" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-accent-400">
              <Cpu className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">5. Machine Learning Models & Architectures</h2>
            </div>
            <p>
              Rather than forcing a single architecture onto heterogeneous vascular targets, CardioVision AI conducted systematic
              hyperparameter and model selection across tree ensembles, gradient boosters, and regularized linear models:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-3.5 rounded-xl bg-surface-darker border border-primary-500/30 space-y-1">
                <span className="text-primary-400 font-bold block">CAD Overall Target: Random Forest</span>
                <p className="text-gray-300 font-sans">Ensemble of 150 de-correlated decision trees with balanced subsampling.</p>
                <span className="text-[11px] text-gray-500 block">Selected Threshold: 0.61 | ROC-AUC: 92.1%</span>
              </div>

              <div className="p-3.5 rounded-xl bg-surface-darker border border-rose-500/30 space-y-1">
                <span className="text-rose-400 font-bold block">LAD Artery Model: LightGBM</span>
                <p className="text-gray-300 font-sans">Histogram-based gradient boosted tree with leaf-wise splitting.</p>
                <span className="text-[11px] text-gray-500 block">Selected Threshold: 0.53 | ROC-AUC: 87.2%</span>
              </div>

              <div className="p-3.5 rounded-xl bg-surface-darker border border-orange-500/30 space-y-1">
                <span className="text-orange-400 font-bold block">LCX Artery Model: XGBoost</span>
                <p className="text-gray-300 font-sans">Extreme gradient boosting with exact greedy split search and L1/L2 shrinkage.</p>
                <span className="text-[11px] text-gray-500 block">Selected Threshold: 0.50 | ROC-AUC: 80.8%</span>
              </div>

              <div className="p-3.5 rounded-xl bg-surface-darker border border-blue-500/30 space-y-1">
                <span className="text-blue-400 font-bold block">RCA Artery Model: Logistic Regression</span>
                <p className="text-gray-300 font-sans">L2-regularized linear model with balanced class weight formulation.</p>
                <span className="text-[11px] text-gray-500 block">Selected Threshold: 0.44 | ROC-AUC: 81.6%</span>
              </div>
            </div>
          </section>

          {/* 6. VALIDATION */}
          <section id="validation" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-yellow-400">
              <CheckCircle2 className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">6. Validation Methodology & Threshold Optimization</h2>
            </div>
            <p>
              Model performance was evaluated using <strong>Stratified 5-Fold Cross-Validation</strong>.
              In stratified cross-validation, the target prevalence ratio (e.g. 71.3% CAD) is rigorously maintained in each of the 5 folds:
            </p>
            <ul className="list-disc pl-5 space-y-1 text-xs font-sans text-gray-300">
              <li><strong>Out-of-Fold (OOF) Prediction:</strong> Predictions are generated strictly on unseen test folds; no training instance is ever evaluated by its own model.</li>
              <li><strong>Threshold Optimization:</strong> Standard 0.50 probability thresholds frequently underperform on imbalanced medical cohorts. Decision thresholds were tuned to maximize the F1-score across out-of-fold predictions.</li>
              <li><strong>Performance Metrics:</strong> Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and ROC-AUC are stored directly in <code className="text-yellow-300 font-mono">metrics.json</code>.</li>
            </ul>
          </section>

          {/* 7. SHAP */}
          <section id="shap" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-yellow-400">
              <Zap className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">7. Explainability via TreeSHAP</h2>
            </div>
            <p>
              To eliminate the "black box" nature of deep tree ensembles, CardioVision AI integrates <strong>TreeSHAP (SHapley Additive exPlanations)</strong>.
              Rooted in cooperative game theory, TreeSHAP calculates the exact marginal contribution of each clinical feature to the
              model's log-odds / probability prediction score:
            </p>
            <div className="p-4 rounded-xl bg-surface-darker border border-surface-border space-y-2 text-xs font-mono">
              <span className="text-yellow-400 font-bold block">Mathematical Additivity:</span>
              <p className="text-gray-300 font-sans">
                The sum of all feature SHAP values plus the base expected model value equals the final model prediction:
                <code className="text-yellow-300 ml-2">f(x) = φ₀ + ∑ φᵢ(x)</code>
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-surface-darker/90 border border-amber-500/40 text-xs font-mono text-amber-200">
              <span className="font-bold text-amber-400 block mb-1">MANDATORY NON-CAUSAL INTERPRETATION PROTOCOL:</span>
              SHAP feature contributions represent <em>statistical influence on the algorithmic score</em> for a specific model.
              SHAP does NOT identify biological causes or pathophysiology. The interface strictly enforces the label
              <strong> "Feature contribution to model output"</strong> and strictly prohibits causal phrasing such as "this feature caused CAD."
            </div>
          </section>

          {/* 8. FASTAPI ARCHITECTURE */}
          <section id="fastapi" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-emerald-400">
              <FileCode2 className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">8. FastAPI Backend Architecture</h2>
            </div>
            <p>
              The backend is built with <strong>FastAPI</strong> and Python, adhering to enterprise asynchronous API standards:
            </p>
            <div className="space-y-2 font-mono text-xs">
              <div className="p-3 rounded-lg bg-surface-darker border border-surface-border flex justify-between items-center">
                <span className="text-emerald-400 font-bold">POST /api/v1/analyze</span>
                <span className="text-gray-400">Unified end-to-end pipeline (predict CAD/LAD/LCX/RCA + TreeSHAP)</span>
              </div>
              <div className="p-3 rounded-lg bg-surface-darker border border-surface-border flex justify-between items-center">
                <span className="text-emerald-400 font-bold">POST /api/v1/predict</span>
                <span className="text-gray-400">Fast inference for all 4 models with tuned classification</span>
              </div>
              <div className="p-3 rounded-lg bg-surface-darker border border-surface-border flex justify-between items-center">
                <span className="text-emerald-400 font-bold">POST /api/v1/explain</span>
                <span className="text-gray-400">Target-specific TreeSHAP feature attributions and ranks</span>
              </div>
              <div className="p-3 rounded-lg bg-surface-darker border border-surface-border flex justify-between items-center">
                <span className="text-blue-400 font-bold">GET /api/v1/metrics/summary</span>
                <span className="text-gray-400">Verified out-of-fold cross-validation metrics directly from artifacts</span>
              </div>
              <div className="p-3 rounded-lg bg-surface-darker border border-surface-border flex justify-between items-center">
                <span className="text-blue-400 font-bold">GET /api/v1/model/info</span>
                <span className="text-gray-400">Model versions, pipeline configurations, and dataset metadata</span>
              </div>
            </div>
            <p className="text-xs text-gray-400">
              Input validation is enforced using <strong>Pydantic v2</strong> schemas with zero-tolerance rejection of leakage columns (<code className="text-red-400 font-mono">Cath, LAD, LCX, RCA</code> in input payload triggers HTTP 422).
            </p>
          </section>

          {/* 9. FRONTEND */}
          <section id="frontend" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-primary-400">
              <Boxes className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">9. Frontend Architecture & Design System</h2>
            </div>
            <p>
              The frontend is crafted using <strong>React 19, TypeScript, Tailwind CSS, and Vite</strong>:
            </p>
            <ul className="list-disc pl-5 space-y-1 text-xs font-sans text-gray-300">
              <li><strong>Global State Store:</strong> <code className="text-primary-300 font-mono">AnalysisContext</code> caches predictions, SHAP explanations across all 4 targets, and clinical patient profiles.</li>
              <li><strong>Design Tokens:</strong> Premium dark mode with HSL-tailored slate backgrounds, glassmorphism (<code className="text-primary-300 font-mono">backdrop-filter: blur(20px)</code>), and accessible typography (Outfit + JetBrains Mono).</li>
              <li><strong>Clinical Presets:</strong> Pre-validated presets (High Risk, Moderate Risk, Low Risk) enable immediate instant testing and reproducibility.</li>
              <li><strong>Print / PDF Support:</strong> Clean CSS print stylesheets format the Report Page into a professional diagnostic summary when printed or saved to PDF.</li>
            </ul>
          </section>

          {/* 10. 3D VISUALIZATION */}
          <section id="visualization-3d" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-accent-400">
              <Compass className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">10. 3D Anatomical Visualization Engine</h2>
            </div>
            <p>
              Interactive 3D anatomical heart visualization is powered by <strong>Three.js</strong>, <code className="text-accent-300 font-mono">@react-three/fiber</code>, and <code className="text-accent-300 font-mono">@react-three/drei</code>:
            </p>
            <div className="p-4 rounded-xl bg-surface-darker border border-surface-border space-y-2 text-xs font-mono">
              <span className="text-accent-400 font-bold block">3D Asset Attribution:</span>
              <p className="text-gray-300 font-sans">
                Anatomical geometry is sourced from <strong>BodyParts3D / Anatomography</strong> by the Database Center for Life Science (DBCLS),
                licensed under Creative Commons Attribution-ShareAlike 2.1 Japan (CC BY-SA 2.1 JP).
              </p>
              <div className="flex items-center gap-2 pt-1 text-gray-400">
                <span>Asset: /models/heart_anatomy_bp3d.glb (Optimized Binary glTF 2.0)</span>
              </div>
            </div>
            <p className="text-xs text-gray-400">
              Features include interactive mouse raycasting, orbit controls with bounded camera azimuth/elevation,
              mesh highlighting on hover, and smooth vessel color transitions based on predicted risk level.
            </p>
          </section>

          {/* 11. VESSEL MAPPING */}
          <section id="vessel-mapping" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-rose-400">
              <Activity className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">11. Vessel-to-Model Probability Risk Mapping</h2>
            </div>
            <p>
              Model outputs are translated into anatomical visual risk states using calibrated risk scale thresholds:
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono text-center">
              <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
                <span className="text-emerald-400 font-bold block">Low Risk</span>
                <span className="text-[10px] text-gray-400">&lt; 30% Risk Score</span>
              </div>
              <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                <span className="text-yellow-400 font-bold block">Moderate Risk</span>
                <span className="text-[10px] text-gray-400">30% – 50% Risk Score</span>
              </div>
              <div className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/30">
                <span className="text-orange-400 font-bold block">High Risk</span>
                <span className="text-[10px] text-gray-400">50% – 70% Risk Score</span>
              </div>
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                <span className="text-red-400 font-bold block">Critical Risk</span>
                <span className="text-[10px] text-gray-400">&gt; 70% Risk Score</span>
              </div>
            </div>
            <p className="text-xs text-gray-400">
              The LAD mesh responds to LAD model probability, LCX mesh to LCX model probability, and RCA mesh to RCA model probability.
              The overall coronary arterial bed reflects global CAD risk score.
            </p>
          </section>

          {/* 12. LIMITATIONS */}
          <section id="limitations" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-amber-400">
              <AlertTriangle className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">12. Methodological & Cohort Limitations</h2>
            </div>
            <div className="space-y-3 text-xs font-sans leading-relaxed text-gray-300">
              <div className="p-4 rounded-xl bg-surface-darker border border-surface-border space-y-1">
                <strong className="text-white block font-mono">1. Modest Dataset Sample Size (N = 303):</strong>
                <p className="text-gray-400">
                  With 303 patient records, statistical confidence intervals around sub-vessel metrics are wider than
                  those in registries with 10,000+ patients. Natural fold variance exists across cross-validation folds.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-surface-darker border border-surface-border space-y-1">
                <strong className="text-white block font-mono">2. Single-Center Tertiary Referral Bias:</strong>
                <p className="text-gray-400">
                  Data was collected from symptomatic patients undergoing catheterization at the Tehran Heart Center.
                  High baseline prevalence (71.3% CAD) restricts direct extrapolation to asymptomatic primary care screening cohorts.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-surface-darker border border-surface-border space-y-1">
                <strong className="text-white block font-mono">3. Lack of Longitudinal Outcome Follow-up:</strong>
                <p className="text-gray-400">
                  The dataset captures cross-sectional angiographic stenosis at catheterization without multi-year follow-up for
                  Major Adverse Cardiac Events (MACE), cardiovascular mortality, or post-stenting patency.
                </p>
              </div>
            </div>
          </section>

          {/* 13. 3D LIMITATION (EXPLICIT REQUIREMENT) */}
          <section id="limitations-3d" className="glass-card p-6 sm:p-8 space-y-4 border-amber-500/40 bg-amber-500/5 scroll-mt-24">
            <div className="flex items-center gap-2.5 text-amber-400">
              <ShieldAlert className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">13. Explicit 3D Anatomical Visualization Limitation</h2>
            </div>
            <div className="p-4 rounded-xl bg-surface-darker/90 border border-amber-500/30 text-amber-200 text-xs font-mono space-y-2">
              <p className="font-bold text-amber-400 text-sm">MANDATORY ANATOMICAL GEOMETRY NOTICE:</p>
              <p className="leading-relaxed">
                The 3D visualization maps model predictions to anatomical vessels.
                It does not show exact lesion coordinates or actual patient-specific anatomical stenosis geometry
                because the tabular dataset does not contain CT/MRI spatial imaging data.
              </p>
            </div>
            <p className="text-xs text-gray-400">
              The 3D heart represents an anatomically accurate standard reference organ (BodyParts3D) onto which statistical
              risk states are projected for intuitive clinician comprehension. It must never be mistaken for patient-specific
              coronary computed tomography angiography (CCTA) segmentation or invasive lumen fluoroscopy.
            </p>
          </section>

          {/* 14. SAFETY */}
          <section id="safety" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-red-400">
              <Shield className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">14. Clinical Safety Governance & Regulatory Classification</h2>
            </div>
            <div className="p-4 rounded-xl bg-surface-darker border border-surface-border text-xs space-y-2 text-gray-300">
              <span className="text-red-400 font-bold font-mono block">Scope of Medical Device Regulation:</span>
              <p className="leading-relaxed">
                CardioVision AI is classified as an <strong>Educational and Clinical Decision-Support Research System</strong>.
                It is <strong>NOT an official medical diagnosis</strong> and is NOT approved by the FDA, CE, or regional regulatory authorities
                for autonomous clinical decision-making or therapy initiation.
              </p>
              <p className="leading-relaxed text-gray-400">
                Healthcare providers must treat algorithmic outputs as supplementary hypothesis generation, corroborating
                predictions against clinical assessment, biomarkers, stress imaging, and diagnostic catheterization.
              </p>
            </div>
          </section>

          {/* 15. REPRODUCIBILITY */}
          <section id="reproducibility" className="glass-card p-6 sm:p-8 space-y-4 border-surface-border scroll-mt-24">
            <div className="flex items-center gap-2.5 text-primary-400">
              <GitBranch className="w-5 h-5 shrink-0" />
              <h2 className="text-xl font-display font-bold text-white">15. Reproducibility & Pipeline Determinism</h2>
            </div>
            <p>
              To ensure 100% scientific reproducibility across research environments:
            </p>
            <ul className="list-disc pl-5 space-y-1 text-xs font-mono text-gray-300">
              <li><strong>Deterministic Random Seeds:</strong> All cross-validation splits, SMOTE resamplers, and model seeds are locked to <code className="text-primary-300">random_state = 42</code>.</li>
              <li><strong>Pre-Trained Artifacts:</strong> Models and preprocessors are serialized as Joblib artifacts in <code className="text-primary-300">backend/artifacts/</code>.</li>
              <li><strong>Automated Test Suite:</strong> Python pytest test suites verify API endpoints, data leakage immunity, TreeSHAP dimensions, and health endpoints.</li>
              <li><strong>Version Pinning:</strong> Python requirements and npm package dependencies are strictly pinned with exact version manifests.</li>
            </ul>
          </section>

        </div>
      </div>
    </main>
  );
}
