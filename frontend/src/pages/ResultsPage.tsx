/**
 * CardioVision AI — Production Results Page (/results)
 * =====================================================
 * Complete multi-vessel clinical prediction dashboard.
 *
 * COMPLIANCE & REQUIREMENTS:
 * 1. Result Cards for: Overall CAD, LAD, LCX, RCA.
 *    Each shows target name, full name, model probability, classification, threshold, explanation access.
 *    NEVER shows probability as "percentage blockage".
 * 2. 3D + Results: Connects active API results with 3D vessel visualization.
 * 3. Vessel Analysis: Detailed section for LAD, LCX, RCA with probability, classification, top features.
 * 4. Physiological Profile: Displays supplied patient measurements alongside model inputs.
 * 5. Model Metrics: Reads real out-of-fold Accuracy, Precision, Recall, F1, ROC-AUC from backend.
 * 6. Responsive: Optimized for desktop, tablet, and mobile.
 */

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAnalysis } from '@/context/AnalysisContext';
import { SafetyDisclaimer } from '@/components/ui/SafetyDisclaimer';
import { PhysiologicalProfileCard } from '@/components/results/PhysiologicalProfileCard';
import { ModelMetricsSection } from '@/components/results/ModelMetricsSection';
import { getRiskLevel, getRiskColor } from '@/types';
import {
  BarChart2,
  Eye,
  Zap,
  AlertTriangle,
  RefreshCw,
  Sparkles,
  ArrowRight,
  Activity,
  SlidersHorizontal,
} from 'lucide-react';

interface ResultTargetConfig {
  id: 'CAD' | 'LAD' | 'LCX' | 'RCA';
  fullName: string;
  clinicalAlias: string;
  territory: string;
  isVessel: boolean;
}

const TARGET_CONFIGS: ResultTargetConfig[] = [
  {
    id: 'CAD',
    fullName: 'Overall Coronary Artery Disease',
    clinicalAlias: 'Primary Catheterization CAD Diagnosis (Cath >= 50% stenosis in any major vessel)',
    territory: 'Multi-vessel global coronary arterial circulation',
    isVessel: false,
  },
  {
    id: 'LAD',
    fullName: 'Left Anterior Descending Artery',
    clinicalAlias: 'Anterior Interventricular Branch of LCA (FMA74912 / FMA3981)',
    territory: 'Anterior myocardial wall, anterior 2/3 interventricular septum, cardiac apex',
    isVessel: true,
  },
  {
    id: 'LCX',
    fullName: 'Left Circumflex Artery',
    clinicalAlias: 'Circumflex Branch of LCA (FMA74923 / FMA3984)',
    territory: 'Posterolateral myocardial wall, obtuse marginal branches, portions of left atrium',
    isVessel: true,
  },
  {
    id: 'RCA',
    fullName: 'Right Coronary Artery',
    clinicalAlias: 'Right Coronary Artery Proper (FMA3802 / FMA3991)',
    territory: 'Right atrium, right ventricle, inferior LV wall, posterior descending branch (85%)',
    isVessel: true,
  },
];

export function ResultsPage() {
  const navigate = useNavigate();
  const {
    predictionResult,
    patientId,
    patientFeatures,
    isLoading,
    runAnalysis,
    fetchExplanation,
  } = useAnalysis();

  const [activeTab, setActiveTab] = useState<'overview' | 'vessels' | 'physiology' | 'metrics'>('overview');

  // Handle empty state gracefully
  if (!predictionResult) {
    return (
      <main id="results-page-empty" className="section-container py-16 animate-fade-in">
        <div className="max-w-2xl mx-auto text-center space-y-6 glass-card p-8 border-surface-border">
          <div className="w-16 h-16 rounded-2xl bg-surface-card border border-surface-border flex items-center justify-center mx-auto text-primary-400">
            <BarChart2 className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-2xl font-display font-bold text-white mb-2">
              No Active Prediction Results
            </h1>
            <p className="text-sm text-gray-400 max-w-md mx-auto leading-relaxed">
              Please enter patient clinical data or load a clinical preset on the Analysis page to compute multi-vessel risk.
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
            <Link
              to="/analyze"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-primary-500 hover:bg-primary-400 text-white font-display font-semibold text-sm transition-all shadow-lg shadow-primary-500/20"
            >
              <span>Go to Clinical Analysis Form</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <button
              onClick={() => runAnalysis()}
              disabled={isLoading}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-surface-dark hover:bg-surface-darker text-gray-200 border border-surface-border font-mono text-sm transition-all"
            >
              <Sparkles className="w-4 h-4 text-accent-400" />
              <span>Run on Default Preset</span>
            </button>
          </div>
        </div>
      </main>
    );
  }

  const predictions = predictionResult.predictions;

  // Compute maximum risk across all vessels
  const cadPred = predictions['CAD'];
  const cadProb = cadPred ? cadPred.model_predicted_probability : 0;
  const overallRiskLevel = getRiskLevel(cadProb, cadPred?.threshold ?? 0.5);
  const overallColor = getRiskColor(overallRiskLevel);

  return (
    <main id="results-page" className="section-container py-8 animate-fade-in space-y-8">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-surface-border pb-6">
        <div>
          <div className="flex items-center gap-2 mb-2 flex-wrap text-xs font-mono">
            <span className="px-2.5 py-0.5 rounded bg-primary-500/15 border border-primary-500/30 text-primary-300 font-bold">
              Patient ID: {patientId}
            </span>
            <span className="px-2.5 py-0.5 rounded bg-surface-dark border border-surface-border text-gray-300">
              Pipeline: {predictionResult.model_version}
            </span>
            <span
              className="px-2.5 py-0.5 rounded font-bold capitalize border"
              style={{
                backgroundColor: `${overallColor}15`,
                borderColor: `${overallColor}40`,
                color: overallColor,
              }}
            >
              CAD Status: {overallRiskLevel} risk
            </span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-display font-black text-white tracking-tight">
            Multi-Vessel Risk Results
          </h1>
          <p className="text-xs text-gray-400 mt-1 max-w-2xl">
            Model-predicted risk scores for overall coronary artery disease and vessel-specific branches.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5 flex-wrap">
          <button
            onClick={() => runAnalysis()}
            disabled={isLoading}
            className="px-4 py-2.5 rounded-xl bg-surface-dark border border-surface-border text-gray-300 hover:text-white text-xs font-mono font-semibold flex items-center gap-2 transition-all"
            title="Re-run POST /analyze"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-accent-400' : ''}`} />
            <span>Re-Run</span>
          </button>

          <Link
            to="/visualization"
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-500 hover:opacity-95 text-white font-display font-bold text-xs flex items-center gap-2 shadow-lg shadow-emerald-500/20 transition-all"
          >
            <Eye className="w-4 h-4" />
            <span>3D Anatomy</span>
          </Link>

          <Link
            to="/insights"
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-amber-600 to-orange-500 hover:opacity-95 text-white font-display font-bold text-xs flex items-center gap-2 shadow-lg shadow-amber-500/20 transition-all"
          >
            <Zap className="w-4 h-4" />
            <span>SHAP Explainability</span>
          </Link>

          <Link
            to="/report"
            className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:opacity-95 text-white font-display font-bold text-xs flex items-center gap-2 shadow-lg shadow-violet-500/20 transition-all"
          >
            <BarChart2 className="w-4 h-4" />
            <span>Generate Report</span>
          </Link>
        </div>
      </div>

      {/* Mandatory Non-Blockage Terminology Banner */}
      <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3 text-xs font-mono">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-0.5 text-amber-200/90 leading-relaxed">
          <p className="font-bold text-amber-300">Model Probability Output Specification</p>
          <p>
            Values represent <strong>model-predicted stenosis risk probabilities</strong> derived from clinical tabular features.
            They are <strong>NEVER percentage blockage</strong> and do NOT represent pixel-level imaging segmentation or lesion coordinates.
          </p>
        </div>
      </div>

      <SafetyDisclaimer />

      {/* Navigation Tabs for Dashboard Views */}
      <div className="flex items-center gap-2 border-b border-surface-border pb-1 overflow-x-auto text-xs font-mono">
        {[
          { id: 'overview', label: '1. Model Result Cards (All 4 Targets)', icon: BarChart2 },
          { id: 'vessels', label: '2. Coronary Vessel Analysis (LAD · LCX · RCA)', icon: Eye },
          { id: 'physiology', label: '3. Physiological Profile', icon: Activity },
          { id: 'metrics', label: '4. Model Evaluation Metrics', icon: SlidersHorizontal },
        ].map((t) => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id as any)}
              className={`px-3 py-2 rounded-lg font-bold flex items-center gap-1.5 whitespace-nowrap transition-all border ${
                isActive
                  ? 'bg-primary-500/20 border-primary-500 text-white shadow'
                  : 'bg-transparent border-transparent text-gray-400 hover:text-gray-200'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{t.label}</span>
            </button>
          );
        })}
      </div>

      {/* VIEW 1: RESULT CARDS (Requirement 1) */}
      {(activeTab === 'overview' || activeTab === 'vessels') && (
        <section id="result-cards-grid" className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-display font-bold text-white flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-primary-400" />
              <span>Model Prediction Cards</span>
            </h2>
            <span className="text-xs font-mono text-gray-400">
              Stratified Out-of-Fold Trained Models
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {TARGET_CONFIGS.map((target) => {
              const pred = predictions[target.id];
              if (!pred) return null;

              const proba = pred.model_predicted_probability;
              const probaPct = (proba * 100).toFixed(1);
              const thresh = pred.threshold;
              const isPositive = pred.classification_binary === 1 || proba >= thresh;
              const riskLevel = getRiskLevel(proba, thresh);
              const riskColor = getRiskColor(riskLevel);

              return (
                <div
                  key={target.id}
                  id={`result-card-${target.id.toLowerCase()}`}
                  className="glass-card p-6 flex flex-col justify-between border-surface-border relative overflow-hidden hover:border-primary-500/40 transition-all duration-300"
                >
                  {/* Background Radial Glow */}
                  <div
                    className="absolute -top-12 -right-12 w-36 h-36 rounded-full blur-3xl pointer-events-none"
                    style={{ backgroundColor: `${riskColor}15` }}
                  />

                  {/* Card Header */}
                  <div>
                    <div className="flex items-center justify-between gap-2 border-b border-surface-border/50 pb-3 mb-4">
                      <div className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full shadow-sm"
                          style={{ backgroundColor: riskColor }}
                        />
                        <span className="font-mono text-xs font-bold text-white">
                          Target: {target.id}
                        </span>
                      </div>
                      <span
                        className="px-2.5 py-0.5 rounded text-xs font-mono font-bold uppercase border"
                        style={{
                          backgroundColor: `${riskColor}15`,
                          borderColor: `${riskColor}30`,
                          color: riskColor,
                        }}
                      >
                        {riskLevel} Risk
                      </span>
                    </div>

                    {/* Name & Alias */}
                    <div className="mb-4">
                      <h3 className="text-xl font-display font-bold text-white">
                        {target.fullName}
                      </h3>
                      <p className="text-xs text-gray-400 mt-0.5 leading-relaxed font-mono">
                        {target.clinicalAlias}
                      </p>
                    </div>

                    {/* Probability Metric Display */}
                    <div className="grid grid-cols-2 gap-3 mb-4 font-mono">
                      <div className="p-3.5 rounded-xl bg-surface-dark/80 border border-surface-border">
                        <span className="text-[10px] text-gray-400 uppercase tracking-wider block mb-1">
                          Model Probability
                        </span>
                        <div className="flex items-baseline gap-1.5">
                          <span className="text-2xl sm:text-3xl font-display font-black text-white">
                            {probaPct}%
                          </span>
                          <span className="text-xs text-gray-400">
                            ({proba.toFixed(4)})
                          </span>
                        </div>
                        <div className="w-full bg-surface-darker h-1.5 rounded-full mt-2 overflow-hidden border border-surface-border/40">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.min(100, proba * 100)}%`,
                              backgroundColor: riskColor,
                            }}
                          />
                        </div>
                      </div>

                      <div className="p-3.5 rounded-xl bg-surface-dark/80 border border-surface-border flex flex-col justify-between">
                        <div>
                          <span className="text-[10px] text-gray-400 uppercase tracking-wider block mb-1">
                            Classification
                          </span>
                          <span
                            className="text-sm font-display font-bold block"
                            style={{ color: riskColor }}
                          >
                            {isPositive ? pred.positive_class_label : 'Normal'}
                          </span>
                        </div>
                        <div className="text-[11px] text-gray-400 mt-2">
                          Decision Threshold: <strong className="text-gray-200">{thresh.toFixed(2)}</strong>
                        </div>
                      </div>
                    </div>

                    {/* Territory info */}
                    <p className="text-xs text-gray-400 font-mono mb-4 leading-relaxed bg-surface-darker/50 p-2.5 rounded-lg border border-surface-border/40">
                      <strong className="text-gray-300">Territory:</strong> {target.territory}
                    </p>
                  </div>

                  {/* Actions / Explanation Access */}
                  <div className="pt-4 border-t border-surface-border/60 flex items-center justify-between gap-2">
                    <button
                      onClick={() => {
                        void fetchExplanation(target.id);
                        navigate('/insights');
                      }}
                      className="px-3 py-1.5 rounded-lg bg-surface-dark hover:bg-surface-darker border border-surface-border text-xs font-mono text-gray-300 hover:text-white flex items-center gap-1.5 transition-all"
                    >
                      <Zap className="w-3.5 h-3.5 text-amber-400" />
                      <span>View SHAP</span>
                    </button>

                    {target.isVessel ? (
                      <Link
                        to="/visualization"
                        className="px-3 py-1.5 rounded-lg bg-primary-500/15 hover:bg-primary-500/25 border border-primary-500/30 text-xs font-mono text-primary-300 hover:text-primary-200 flex items-center gap-1.5 transition-all"
                      >
                        <Eye className="w-3.5 h-3.5 text-emerald-400" />
                        <span>Inspect in 3D</span>
                      </Link>
                    ) : (
                      <span className="text-[11px] font-mono text-gray-500">
                        Primary Diagnosis
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* VIEW 2: CORONARY VESSEL ANALYSIS (Requirement 3) */}
      {(activeTab === 'overview' || activeTab === 'vessels') && (
        <section id="coronary-vessel-analysis-section" className="glass-card p-6 border-surface-border space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-surface-border/60 pb-3">
            <div>
              <h2 className="text-xl font-display font-bold text-white flex items-center gap-2">
                <Eye className="w-5 h-5 text-emerald-400" />
                <span>Coronary Vessel Detailed Analysis</span>
              </h2>
              <p className="text-xs text-gray-400">
                LAD, LCX, and RCA vessel-specific probabilities, clinical classifications, and top contributing features.
              </p>
            </div>
            <Link
              to="/visualization"
              className="px-3.5 py-1.5 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 hover:text-white text-xs font-mono font-bold flex items-center gap-1.5 self-start sm:self-auto transition-all"
            >
              <span>Open 3D Viewer</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {(['LAD', 'LCX', 'RCA'] as const).map((vKey) => {
              const pred = predictions[vKey];
              if (!pred) return null;
              const proba = pred.model_predicted_probability;
              const thresh = pred.threshold;
              const isStenotic = pred.classification_binary === 1 || proba >= thresh;
              const riskColor = getRiskColor(getRiskLevel(proba, thresh));
              const title =
                vKey === 'LAD'
                  ? 'Left Anterior Descending'
                  : vKey === 'LCX'
                  ? 'Left Circumflex'
                  : 'Right Coronary Artery';

              return (
                <div
                  key={vKey}
                  className="p-4 rounded-xl bg-surface-dark/70 border border-surface-border/80 flex flex-col justify-between space-y-3"
                >
                  <div>
                    <div className="flex items-center justify-between border-b border-surface-border/40 pb-2 mb-2">
                      <span className="font-mono text-xs font-extrabold text-white">
                        {vKey}
                      </span>
                      <span
                        className="text-xs font-mono font-bold"
                        style={{ color: riskColor }}
                      >
                        {(proba * 100).toFixed(1)}% Prob
                      </span>
                    </div>

                    <h4 className="text-sm font-display font-bold text-gray-200 mb-1">
                      {title}
                    </h4>
                    <div className="text-[11px] font-mono text-gray-400 mb-3">
                      Classification:{' '}
                      <strong style={{ color: riskColor }}>
                        {isStenotic ? 'Stenotic (High Risk)' : 'Normal (Low Risk)'}
                      </strong>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-surface-darker h-2 rounded-full overflow-hidden border border-surface-border/30 mb-3">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, proba * 100)}%`, backgroundColor: riskColor }}
                      />
                    </div>
                  </div>

                  <Link
                    to="/visualization"
                    className="w-full py-2 rounded-lg bg-surface-darker hover:bg-surface-card border border-surface-border/60 text-center text-xs font-mono text-gray-300 hover:text-white transition-all flex items-center justify-center gap-1.5"
                  >
                    <span>Inspect Vessel in 3D</span>
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* VIEW 3: PHYSIOLOGICAL PROFILE (Requirement 5) */}
      {(activeTab === 'overview' || activeTab === 'physiology') && (
        <PhysiologicalProfileCard
          patientFeatures={patientFeatures}
          patientId={patientId}
        />
      )}

      {/* VIEW 4: MODEL EVALUATION METRICS (Requirement 6) */}
      {(activeTab === 'overview' || activeTab === 'metrics') && (
        <ModelMetricsSection />
      )}
    </main>
  );
}
