/**
 * CardioVision AI — Advanced SHAP Explainability Dashboard (/insights)
 * ======================================================================
 * Enterprise-grade interpretability dashboard for multi-vessel CAD risk.
 * Provides interactive TreeSHAP feature attribution visualizations for:
 *   - CAD (Overall Cath diagnosis model)
 *   - LAD (Left Anterior Descending artery model)
 *   - LCX (Left Circumflex artery model)
 *   - RCA (Right Coronary Artery model)
 *
 * Core Features:
 *   - Top positive contributors (+SHAP, pushing model risk probability higher)
 *   - Top negative contributors (-SHAP, pushing model risk probability lower)
 *   - Absolute contributions (|SHAP|)
 *   - Raw patient value & sign-coded SHAP attribution
 *   - Feature categories strictly mapped from UCI ID 411:
 *       • Demographic (Age, Weight, Length, Sex, BMI)
 *       • Clinical (Medical history, Symptoms, Signs, Physical exam)
 *       • Laboratory (FBS, Lipids, Renal, CBC, Electrolytes)
 *       • ECG (Q Wave, ST changes, T inversion, LVH, Arrhythmia/BBB)
 *       • Echo (EF-TTE, Region RWMA)
 *   - Category filtering & Category Domain Impact Breakdown
 *   - Strict non-causal language: "Feature contribution to model output"
 *     (Never implies clinical causality or disease etiology)
 */

import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useAnalysis } from '@/context/AnalysisContext';
import { SafetyDisclaimer } from '@/components/ui/SafetyDisclaimer';
import { ShapBarChart } from '@/components/ui/ShapBarChart';
import {
  getFeatureCategory,
  CATEGORY_STYLES,
  MODEL_CONTRIBUTION_LABEL,
  NON_CAUSAL_DISCLAIMER,
  type FeatureCategory,
} from '@/utils';
import type { FeatureContribution } from '@/types';
import {
  Zap,
  TrendingUp,
  TrendingDown,
  Search,
  AlertCircle,
  Loader2,
  ShieldAlert,
  BarChart2,
  ArrowRight,
  Info,
  Filter,
  Layers,
  Activity,
  HeartPulse,
  Eye,
} from 'lucide-react';

const TARGET_INFO: Record<string, { label: string; vesselName: string; color: string; bg: string; border: string; desc: string }> = {
  CAD: {
    label: 'CAD (Overall)',
    vesselName: 'Cath Diagnosis Target',
    color: 'text-primary-400',
    bg: 'bg-primary-500/10',
    border: 'border-primary-500',
    desc: 'Primary Coronary Artery Disease diagnosis model (Random Forest, tuned threshold 0.61)',
  },
  LAD: {
    label: 'LAD Artery',
    vesselName: 'Left Anterior Descending',
    color: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500',
    desc: 'Left Anterior Descending stenosis model (LightGBM, tuned threshold 0.53)',
  },
  LCX: {
    label: 'LCX Artery',
    vesselName: 'Left Circumflex',
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500',
    desc: 'Left Circumflex stenosis model (XGBoost, tuned threshold 0.50)',
  },
  RCA: {
    label: 'RCA Artery',
    vesselName: 'Right Coronary Artery',
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500',
    desc: 'Right Coronary Artery stenosis model (Logistic Regression, tuned threshold 0.44)',
  },
};

const CATEGORIES: Array<{ key: 'ALL' | FeatureCategory; label: string; count: number }> = [
  { key: 'ALL', label: 'All Categories', count: 55 },
  { key: 'Clinical', label: 'Clinical', count: 26 },
  { key: 'Laboratory', label: 'Laboratory', count: 14 },
  { key: 'ECG', label: 'ECG', count: 8 },
  { key: 'Demographic', label: 'Demographic', count: 5 },
  { key: 'Echo', label: 'Echo', count: 2 },
];

export function InsightsPage() {
  const {
    explainResult,
    explainResults,
    activeExplainTarget,
    fetchExplanation,
    predictionResult,
    isLoading,
    error,
  } = useAnalysis();

  const [selectedTarget, setSelectedTarget] = useState<string>(activeExplainTarget || 'CAD');
  const [selectedCategory, setSelectedCategory] = useState<'ALL' | FeatureCategory>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [directionFilter, setDirectionFilter] = useState<'ALL' | 'positive' | 'negative'>('ALL');
  const [activeView, setActiveView] = useState<'chart' | 'cards' | 'table' | 'categories'>('chart');

  // Sync selected target with active target from context
  useEffect(() => {
    if (activeExplainTarget && activeExplainTarget !== selectedTarget) {
      setSelectedTarget(activeExplainTarget);
    }
  }, [activeExplainTarget, selectedTarget]);

  // Load explanation on target change if not loaded
  useEffect(() => {
    if (!explainResults[selectedTarget] && !isLoading) {
      void fetchExplanation(selectedTarget).catch(() => {});
    }
  }, [selectedTarget, explainResults, isLoading, fetchExplanation]);

  const currentExplain = explainResults[selectedTarget] || explainResult;

  // Enrich raw contributions with dataset category
  const enrichedContributions = useMemo<FeatureContribution[]>(() => {
    if (!currentExplain?.all_contributions) return [];
    return currentExplain.all_contributions.map((item) => ({
      ...item,
      feature_category: item.feature_category || getFeatureCategory(item.feature),
    }));
  }, [currentExplain]);

  // Filtered contributions by Category, Direction & Search
  const filteredContributions = useMemo(() => {
    return enrichedContributions.filter((item) => {
      // Category filter
      if (selectedCategory !== 'ALL' && item.feature_category !== selectedCategory) {
        return false;
      }
      // Direction filter
      if (directionFilter === 'positive' && item.shap_value <= 0) {
        return false;
      }
      if (directionFilter === 'negative' && item.shap_value >= 0) {
        return false;
      }
      // Search text
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesName = item.feature.toLowerCase().includes(q);
        const matchesCategory = item.feature_category?.toLowerCase().includes(q);
        const matchesVal = String(item.patient_value).toLowerCase().includes(q);
        if (!matchesName && !matchesCategory && !matchesVal) return false;
      }
      return true;
    });
  }, [enrichedContributions, selectedCategory, directionFilter, searchQuery]);

  // Top positive and negative contributors from enriched list
  const topPositive = useMemo(() => {
    return enrichedContributions
      .filter((item) => item.shap_value > 0)
      .sort((a, b) => b.shap_value - a.shap_value);
  }, [enrichedContributions]);

  const topNegative = useMemo(() => {
    return enrichedContributions
      .filter((item) => item.shap_value < 0)
      .sort((a, b) => a.shap_value - b.shap_value);
  }, [enrichedContributions]);

  // Category aggregates: sum of absolute contributions
  const categoryStats = useMemo(() => {
    const stats: Record<FeatureCategory, { totalAbs: number; count: number; maxAbs: number }> = {
      Demographic: { totalAbs: 0, count: 0, maxAbs: 0 },
      Clinical: { totalAbs: 0, count: 0, maxAbs: 0 },
      Laboratory: { totalAbs: 0, count: 0, maxAbs: 0 },
      ECG: { totalAbs: 0, count: 0, maxAbs: 0 },
      Echo: { totalAbs: 0, count: 0, maxAbs: 0 },
    };

    enrichedContributions.forEach((item) => {
      const cat = (item.feature_category as FeatureCategory) || getFeatureCategory(item.feature);
      if (stats[cat]) {
        stats[cat].totalAbs += item.absolute_contribution;
        stats[cat].count += 1;
        if (item.absolute_contribution > stats[cat].maxAbs) {
          stats[cat].maxAbs = item.absolute_contribution;
        }
      }
    });

    const totalSystemAbs = Object.values(stats).reduce((acc, c) => acc + c.totalAbs, 0) || 1;
    return { stats, totalSystemAbs };
  }, [enrichedContributions]);

  const targetStyle = TARGET_INFO[selectedTarget] ?? TARGET_INFO.CAD;

  // Max absolute contribution for relative impact bars
  const maxContribution = useMemo(() => {
    if (!enrichedContributions.length) return 1;
    return Math.max(...enrichedContributions.map((c) => c.absolute_contribution), 0.001);
  }, [enrichedContributions]);

  // Empty state when no analysis run yet
  if (!predictionResult && !currentExplain && !isLoading) {
    return (
      <main id="insights-page-empty" className="section-container py-16 animate-fade-in">
        <div className="max-w-xl mx-auto text-center space-y-6 glass-card p-10 border-surface-border">
          <div className="w-16 h-16 rounded-2xl bg-yellow-500/15 border border-yellow-500/30 flex items-center justify-center mx-auto shadow-glow-accent">
            <Zap className="w-8 h-8 text-yellow-400" />
          </div>
          <div>
            <h1 className="text-2xl font-display font-bold text-white mb-2">No Analysis Results Available</h1>
            <p className="text-sm text-gray-400 leading-relaxed">
              Run a patient analysis or select one of the clinical presets to compute real TreeSHAP feature attributions
              across CAD, LAD, LCX, and RCA models.
            </p>
          </div>
          <div className="pt-2 flex justify-center">
            <Link
              to="/analyze"
              className="btn-accent px-6 py-3 text-sm flex items-center gap-2"
            >
              <Activity className="w-4 h-4" />
              <span>Go to Clinical Analysis</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main id="insights-page" className="section-container py-10 animate-fade-in space-y-8">
      {/* ── Page Header ─────────────────────────────────────── */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-yellow-500/15 border border-yellow-500/30 text-yellow-400 text-xs font-mono font-medium">
          <Zap className="w-3.5 h-3.5" />
          <span>TreeSHAP Multi-Vessel Interpretability</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-display font-bold text-white tracking-tight">
          Advanced SHAP Explainability Dashboard
        </h1>
        <p className="text-gray-400 text-sm leading-relaxed max-w-2xl mx-auto">
          Quantifies algorithmic feature attributions for CAD diagnosis and multi-vessel stenosis models.
          Filter by anatomical vessel, clinical domain category, and attribution direction.
        </p>
      </div>

      <SafetyDisclaimer />

      {/* ── Target Model Selector Bar ──────────────────────── */}
      <div className="glass-card p-4 sm:p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 w-full md:w-auto">
          <span className="text-xs font-mono text-gray-400 uppercase tracking-wider shrink-0 flex items-center gap-1.5">
            <HeartPulse className="w-3.5 h-3.5 text-accent-400" />
            <span>Target Model:</span>
          </span>
          <div className="grid grid-cols-2 sm:flex sm:flex-wrap items-center gap-2 w-full sm:w-auto">
            {(['CAD', 'LAD', 'LCX', 'RCA'] as const).map((target) => {
              const info = TARGET_INFO[target];
              const isActive = selectedTarget === target;
              const proba = predictionResult?.predictions?.[target]?.model_predicted_probability;
              return (
                <button
                  key={target}
                  id={`shap-target-${target.toLowerCase()}`}
                  onClick={() => setSelectedTarget(target)}
                  className={`px-3.5 py-2 rounded-xl text-xs font-mono font-bold transition-all border flex items-center justify-between sm:justify-start gap-2 ${
                    isActive
                      ? `${info.bg} ${info.border} ${info.color} shadow-lg shadow-${target.toLowerCase()}-500/10`
                      : 'bg-surface-dark/80 border-surface-border text-gray-400 hover:text-white hover:border-gray-600'
                  }`}
                >
                  <span>{info.label}</span>
                  {proba !== undefined && (
                    <span className="text-[11px] px-1.5 py-0.2 rounded bg-surface-darker/80 opacity-90">
                      {(proba * 100).toFixed(0)}%
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Model Contribution Language Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-darker/90 border border-surface-border text-xs font-mono text-yellow-400 self-stretch md:self-auto justify-center">
          <ShieldAlert className="w-3.5 h-3.5 text-yellow-400 shrink-0" />
          <span>Label: {MODEL_CONTRIBUTION_LABEL}</span>
        </div>
      </div>

      {/* ── Strict Non-Causal Guidance Notice ──────────────── */}
      <div className="p-4 rounded-xl bg-surface-darker/90 border border-amber-500/30 flex items-start gap-3.5 text-xs font-mono">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="text-gray-200 font-semibold">
            Interpretability Notice — Statistical Feature Attribution Only
          </p>
          <p className="text-gray-400 font-sans leading-relaxed">
            {NON_CAUSAL_DISCLAIMER}
          </p>
        </div>
      </div>

      {/* ── Loading State ───────────────────────────────────── */}
      {isLoading && (
        <div className="glass-card p-12 text-center text-gray-400 flex flex-col items-center justify-center space-y-3">
          <Loader2 className="w-8 h-8 animate-spin text-yellow-400" />
          <p className="font-mono text-xs text-white">Computing TreeSHAP Feature Attributions for {selectedTarget}…</p>
          <p className="text-[11px] text-gray-500">Transforming 55 clinical features and querying trained tree ensemble explainer.</p>
        </div>
      )}

      {/* ── Error Banner ────────────────────────────────────── */}
      {error && !isLoading && (
        <div className="p-4 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* ── Dashboard Content ───────────────────────────────── */}
      {currentExplain && !isLoading && (
        <>
          {/* Target Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="glass-card p-4 space-y-1">
              <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider block">Target Vessel Model</span>
              <p className={`text-xl font-display font-bold ${targetStyle.color}`}>{targetStyle.label}</p>
              <p className="text-[11px] text-gray-400 leading-snug">{targetStyle.desc}</p>
            </div>

            <div className="glass-card p-4 space-y-1">
              <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider block">Model-Predicted Probability</span>
              <p className="text-xl font-display font-bold text-yellow-400">
                {(currentExplain.model_predicted_probability * 100).toFixed(1)}%
              </p>
              <p className="text-[11px] text-gray-400 font-mono">
                Base Value: {currentExplain.base_value.toFixed(4)}
              </p>
            </div>

            <div className="glass-card p-4 space-y-1">
              <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider block">Top Positive Factor</span>
              {topPositive[0] ? (
                <>
                  <p className="text-sm font-semibold text-white truncate">{topPositive[0].feature}</p>
                  <p className="text-[11px] font-mono text-red-400 font-bold">
                    +{topPositive[0].shap_value.toFixed(4)} (Val: {String(topPositive[0].patient_value)})
                  </p>
                </>
              ) : (
                <p className="text-xs text-gray-500">None detected</p>
              )}
            </div>

            <div className="glass-card p-4 space-y-1">
              <span className="text-[11px] font-mono text-gray-400 uppercase tracking-wider block">Top Protective Factor</span>
              {topNegative[0] ? (
                <>
                  <p className="text-sm font-semibold text-white truncate">{topNegative[0].feature}</p>
                  <p className="text-[11px] font-mono text-emerald-400 font-bold">
                    {topNegative[0].shap_value.toFixed(4)} (Val: {String(topNegative[0].patient_value)})
                  </p>
                </>
              ) : (
                <p className="text-xs text-gray-500">None detected</p>
              )}
            </div>
          </div>

          {/* ── Category Domain Impact Breakdown ──────────────── */}
          <div className="glass-card p-5 border-surface-border space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-surface-border/60 pb-3">
              <div>
                <h3 className="text-sm font-display font-bold text-white flex items-center gap-2">
                  <Layers className="w-4 h-4 text-primary-400" />
                  <span>Clinical Category Impact Breakdown (|SHAP| Sum)</span>
                </h3>
                <p className="text-xs text-gray-400">
                  Relative contribution of each clinical category domain to {selectedTarget} model output.
                </p>
              </div>
              <span className="text-xs font-mono text-gray-500">
                55 Features across 5 Dataset Domains
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
              {(['Demographic', 'Clinical', 'Laboratory', 'ECG', 'Echo'] as const).map((cat) => {
                const style = CATEGORY_STYLES[cat];
                const data = categoryStats.stats[cat];
                const pct = ((data.totalAbs / categoryStats.totalSystemAbs) * 100).toFixed(1);
                const isSelected = selectedCategory === cat;

                return (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(isSelected ? 'ALL' : cat)}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      isSelected
                        ? `${style.bg} ${style.border} ring-1 ring-yellow-400/50`
                        : 'bg-surface-dark/70 border-surface-border hover:border-gray-600'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className={`text-[10px] font-mono font-bold uppercase tracking-wider ${style.text}`}>
                        {cat}
                      </span>
                      <span className="text-[10px] font-mono text-gray-400">{data.count} feats</span>
                    </div>
                    <div className="text-lg font-display font-bold text-white mb-1">
                      {pct}%
                    </div>
                    <div className="w-full h-1.5 rounded-full bg-surface-darker overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-accent-400 to-primary-400"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-gray-400 mt-1 block">
                      |SHAP| sum: {data.totalAbs.toFixed(3)}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* ── Filter & View Mode Controls ───────────────────── */}
          <div className="glass-card p-4 space-y-4">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              {/* Category Filter Pills */}
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-xs font-mono text-gray-400 mr-1 flex items-center gap-1">
                  <Filter className="w-3.5 h-3.5 text-gray-400" />
                  <span>Category:</span>
                </span>
                {CATEGORIES.map(({ key, label }) => {
                  const isActive = selectedCategory === key;
                  const count =
                    key === 'ALL'
                      ? enrichedContributions.length
                      : enrichedContributions.filter((c) => c.feature_category === key).length;
                  return (
                    <button
                      key={key}
                      onClick={() => setSelectedCategory(key)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all border ${
                        isActive
                          ? 'bg-yellow-500/20 border-yellow-500 text-yellow-300 font-bold'
                          : 'bg-surface-dark border-surface-border text-gray-400 hover:text-white'
                      }`}
                    >
                      {label} ({count})
                    </button>
                  );
                })}
              </div>

              {/* View Selector */}
              <div className="flex items-center gap-1 bg-surface-darker p-1 rounded-xl border border-surface-border self-start lg:self-auto">
                {([
                  { id: 'chart', label: 'Bar Chart', icon: BarChart2 },
                  { id: 'cards', label: 'Top Contributors', icon: TrendingUp },
                  { id: 'table', label: 'Ranked Table', icon: Search },
                  { id: 'categories', label: 'Category View', icon: Layers },
                ] as const).map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => setActiveView(id)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeView === id
                        ? 'bg-yellow-500/20 text-yellow-300 font-bold shadow-sm'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Sub-Filters: Search & Direction */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-3 border-t border-surface-border/50">
              <div className="relative w-full sm:w-72">
                <Search className="w-3.5 h-3.5 text-gray-400 absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Search feature or patient value..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-surface-dark border border-surface-border rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-gray-500 focus:border-yellow-400 outline-none font-mono"
                />
              </div>

              <div className="flex items-center gap-2 w-full sm:w-auto justify-end text-xs font-mono">
                <span className="text-gray-400">Direction:</span>
                {(['ALL', 'positive', 'negative'] as const).map((d) => (
                  <button
                    key={d}
                    onClick={() => setDirectionFilter(d)}
                    className={`px-2.5 py-1 rounded text-xs transition-colors ${
                      directionFilter === d
                        ? d === 'positive'
                          ? 'bg-red-500/20 text-red-300 font-bold border border-red-500/40'
                          : d === 'negative'
                          ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40'
                          : 'bg-surface-dark text-white font-bold border border-surface-border'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    {d === 'ALL' ? 'All Signs' : d === 'positive' ? '+ Risk' : '- Protective'}
                  </button>
                ))}
                <span className="text-gray-500 text-[11px] ml-2">
                  Showing {filteredContributions.length} of {enrichedContributions.length}
                </span>
              </div>
            </div>
          </div>

          {/* ── VIEW 1: HORIZONTAL BAR CHART ──────────────────── */}
          {activeView === 'chart' && (
            <div className="glass-card p-6 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-surface-border pb-4">
                <div>
                  <h3 className="text-lg font-display font-bold text-white flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-accent-400" />
                    <span>TreeSHAP Waterfall — Top Feature Contributions</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Ranked by |SHAP| impact for {targetStyle.label} output.
                    {selectedCategory !== 'ALL' && ` Filtered to category: ${selectedCategory}.`}
                  </p>
                </div>
                <div className="flex items-center gap-3 text-xs font-mono">
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-2 rounded bg-red-500" />
                    <span className="text-gray-300">Positive (+SHAP)</span>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-2 rounded bg-emerald-500" />
                    <span className="text-gray-300">Negative (-SHAP)</span>
                  </span>
                </div>
              </div>

              {filteredContributions.length > 0 ? (
                <ShapBarChart
                  contributions={filteredContributions}
                  topN={15}
                  title=""
                />
              ) : (
                <div className="py-12 text-center text-gray-500 text-xs font-mono">
                  No features match the selected category and search filters.
                </div>
              )}
            </div>
          )}

          {/* ── VIEW 2: SPLIT TOP CONTRIBUTORS CARDS ──────────── */}
          {activeView === 'cards' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Positive Risk Factors (+SHAP) */}
              <div className="glass-card p-6 border-red-500/20 bg-red-500/5 space-y-4">
                <div className="flex items-center justify-between border-b border-surface-border/70 pb-3">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-red-400" />
                    <h3 className="font-display font-bold text-white">Top Positive Contributors (+SHAP)</h3>
                  </div>
                  <span className="text-xs font-mono text-red-300 font-semibold">
                    Increases Predicted Risk
                  </span>
                </div>

                <div className="space-y-3">
                  {topPositive.slice(0, 10).map((item, idx) => {
                    const cat = (item.feature_category as FeatureCategory) || getFeatureCategory(item.feature);
                    const catStyle = CATEGORY_STYLES[cat] || CATEGORY_STYLES.Clinical;
                    const relPct = Math.min(100, (item.absolute_contribution / maxContribution) * 100);

                    return (
                      <div
                        key={item.feature}
                        className="p-3.5 rounded-xl bg-surface-dark/80 border border-surface-border hover:border-red-500/40 transition-colors space-y-2"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="w-5 h-5 rounded-md bg-red-500/20 text-red-400 text-[10px] font-mono font-bold flex items-center justify-center shrink-0">
                              {idx + 1}
                            </span>
                            <div className="min-w-0">
                              <p className="text-xs font-bold text-white truncate">{item.feature}</p>
                              <span className={`inline-block px-1.5 py-0.2 rounded text-[9px] font-mono border ${catStyle.badge}`}>
                                {cat}
                              </span>
                            </div>
                          </div>

                          <div className="text-right shrink-0">
                            <span className="px-2 py-0.5 rounded bg-red-500/20 border border-red-500/30 text-red-300 font-mono text-xs font-bold">
                              +{item.shap_value.toFixed(4)}
                            </span>
                            <span className="block text-[10px] text-gray-400 font-mono mt-0.5">
                              |Impact|: {item.absolute_contribution.toFixed(4)}
                            </span>
                          </div>
                        </div>

                        {/* Relative impact bar */}
                        <div className="w-full h-1 bg-surface-darker rounded-full overflow-hidden">
                          <div className="h-full bg-red-500 rounded-full" style={{ width: `${relPct}%` }} />
                        </div>

                        <div className="flex justify-between items-center text-[11px] font-mono text-gray-400 pt-1">
                          <span>Patient Value: <strong className="text-gray-200">{String(item.patient_value)}</strong></span>
                          <span className="text-[10px] text-gray-500">{MODEL_CONTRIBUTION_LABEL}</span>
                        </div>
                      </div>
                    );
                  })}

                  {topPositive.length === 0 && (
                    <div className="py-8 text-center text-xs text-gray-500 font-mono">
                      No positive contributors for this target and filter.
                    </div>
                  )}
                </div>
              </div>

              {/* Negative Factors (-SHAP) */}
              <div className="glass-card p-6 border-emerald-500/20 bg-emerald-500/5 space-y-4">
                <div className="flex items-center justify-between border-b border-surface-border/70 pb-3">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-5 h-5 text-emerald-400" />
                    <h3 className="font-display font-bold text-white">Top Negative Contributors (-SHAP)</h3>
                  </div>
                  <span className="text-xs font-mono text-emerald-300 font-semibold">
                    Decreases Predicted Risk
                  </span>
                </div>

                <div className="space-y-3">
                  {topNegative.slice(0, 10).map((item, idx) => {
                    const cat = (item.feature_category as FeatureCategory) || getFeatureCategory(item.feature);
                    const catStyle = CATEGORY_STYLES[cat] || CATEGORY_STYLES.Clinical;
                    const relPct = Math.min(100, (item.absolute_contribution / maxContribution) * 100);

                    return (
                      <div
                        key={item.feature}
                        className="p-3.5 rounded-xl bg-surface-dark/80 border border-surface-border hover:border-emerald-500/40 transition-colors space-y-2"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="w-5 h-5 rounded-md bg-emerald-500/20 text-emerald-400 text-[10px] font-mono font-bold flex items-center justify-center shrink-0">
                              {idx + 1}
                            </span>
                            <div className="min-w-0">
                              <p className="text-xs font-bold text-white truncate">{item.feature}</p>
                              <span className={`inline-block px-1.5 py-0.2 rounded text-[9px] font-mono border ${catStyle.badge}`}>
                                {cat}
                              </span>
                            </div>
                          </div>

                          <div className="text-right shrink-0">
                            <span className="px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 font-mono text-xs font-bold">
                              {item.shap_value.toFixed(4)}
                            </span>
                            <span className="block text-[10px] text-gray-400 font-mono mt-0.5">
                              |Impact|: {item.absolute_contribution.toFixed(4)}
                            </span>
                          </div>
                        </div>

                        {/* Relative impact bar */}
                        <div className="w-full h-1 bg-surface-darker rounded-full overflow-hidden">
                          <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${relPct}%` }} />
                        </div>

                        <div className="flex justify-between items-center text-[11px] font-mono text-gray-400 pt-1">
                          <span>Patient Value: <strong className="text-gray-200">{String(item.patient_value)}</strong></span>
                          <span className="text-[10px] text-gray-500">{MODEL_CONTRIBUTION_LABEL}</span>
                        </div>
                      </div>
                    );
                  })}

                  {topNegative.length === 0 && (
                    <div className="py-8 text-center text-xs text-gray-500 font-mono">
                      No protective contributors for this target and filter.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ── VIEW 3: FULL RANKED TABLE ──────────────────────── */}
          {activeView === 'table' && (
            <div className="glass-card p-6 space-y-4">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-surface-border pb-4">
                <div>
                  <h3 className="text-lg font-display font-bold text-white flex items-center gap-2">
                    <Search className="w-5 h-5 text-accent-400" />
                    <span>Complete Feature Contribution Table</span>
                  </h3>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Showing {filteredContributions.length} features ranked by absolute SHAP attribution (|SHAP|)
                  </p>
                </div>
                <span className="text-xs font-mono text-gray-500">
                  Target: {selectedTarget}
                </span>
              </div>

              <div className="overflow-x-auto scrollbar-thin">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-surface-border text-gray-400 font-mono bg-surface-dark/50">
                      <th className="py-3 px-3">Rank</th>
                      <th className="py-3 px-3">Feature Name</th>
                      <th className="py-3 px-3">Category</th>
                      <th className="py-3 px-3">Patient Value</th>
                      <th className="py-3 px-3">SHAP Value</th>
                      <th className="py-3 px-3">|Contribution|</th>
                      <th className="py-3 px-3">Direction</th>
                      <th className="py-3 px-3 w-28">Relative Impact</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-border/50">
                    {filteredContributions.map((item) => {
                      const cat = (item.feature_category as FeatureCategory) || getFeatureCategory(item.feature);
                      const catStyle = CATEGORY_STYLES[cat] || CATEGORY_STYLES.Clinical;
                      const relPct = Math.min(100, (item.absolute_contribution / maxContribution) * 100);

                      return (
                        <tr key={item.feature} className="hover:bg-surface-card/60 font-mono transition-colors">
                          <td className="py-2.5 px-3 text-gray-400">#{item.rank}</td>
                          <td className="py-2.5 px-3 font-sans font-medium text-white">{item.feature}</td>
                          <td className="py-2.5 px-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${catStyle.badge}`}>
                              {cat}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-gray-200">{String(item.patient_value)}</td>
                          <td className={`py-2.5 px-3 font-bold ${
                            item.shap_value > 0 ? 'text-red-400' : item.shap_value < 0 ? 'text-emerald-400' : 'text-gray-400'
                          }`}>
                            {item.shap_value > 0 ? `+${item.shap_value.toFixed(6)}` : item.shap_value.toFixed(6)}
                          </td>
                          <td className="py-2.5 px-3 text-gray-100 font-semibold">{item.absolute_contribution.toFixed(6)}</td>
                          <td className="py-2.5 px-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${
                              item.direction === 'positive'
                                ? 'bg-red-500/15 text-red-400 border border-red-500/30'
                                : item.direction === 'negative'
                                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                                : 'bg-gray-500/15 text-gray-400 border border-gray-500/30'
                            }`}>
                              {item.direction}
                            </span>
                          </td>
                          <td className="py-2.5 px-3">
                            <div className="w-24 h-1.5 rounded-full bg-surface-darker overflow-hidden">
                              <div
                                className={`h-full rounded-full ${item.shap_value > 0 ? 'bg-red-500' : 'bg-emerald-500'}`}
                                style={{ width: `${relPct}%` }}
                              />
                            </div>
                          </td>
                        </tr>
                      );
                    })}

                    {filteredContributions.length === 0 && (
                      <tr>
                        <td colSpan={8} className="py-8 text-center text-gray-500 text-xs font-mono">
                          No features match "{searchQuery}"
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── VIEW 4: CATEGORY DEEP-DIVE ────────────────────── */}
          {activeView === 'categories' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {(['Demographic', 'Clinical', 'Laboratory', 'ECG', 'Echo'] as const).map((cat) => {
                  const style = CATEGORY_STYLES[cat];
                  const feats = enrichedContributions.filter((c) => c.feature_category === cat);
                  const totalAbs = feats.reduce((sum, c) => sum + c.absolute_contribution, 0);

                  return (
                    <div
                      key={cat}
                      className={`glass-card p-5 border ${style.border} space-y-4 flex flex-col justify-between`}
                    >
                      <div>
                        <div className="flex items-center justify-between border-b border-surface-border pb-3 mb-3">
                          <span className={`px-2.5 py-1 rounded text-xs font-mono font-bold border ${style.badge}`}>
                            {cat} Domain
                          </span>
                          <span className="text-xs font-mono text-gray-400">
                            {feats.length} Features
                          </span>
                        </div>

                        <div className="flex justify-between items-baseline mb-3">
                          <span className="text-xs text-gray-400 font-mono">Total Category Impact:</span>
                          <span className="text-base font-display font-bold text-white">
                            {totalAbs.toFixed(4)}
                          </span>
                        </div>

                        <div className="space-y-2">
                          <span className="text-[10px] uppercase font-mono text-gray-500 block">Top Contributors:</span>
                          {feats.slice(0, 4).map((f) => (
                            <div
                              key={f.feature}
                              className="p-2 rounded-lg bg-surface-dark/70 border border-surface-border text-xs font-mono flex items-center justify-between"
                            >
                              <span className="text-gray-300 font-sans truncate mr-2">{f.feature}</span>
                              <span className={`font-bold shrink-0 ${f.shap_value > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                {f.shap_value > 0 ? `+${f.shap_value.toFixed(4)}` : f.shap_value.toFixed(4)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <button
                        onClick={() => {
                          setSelectedCategory(cat);
                          setActiveView('table');
                        }}
                        className="w-full mt-3 py-2 rounded-lg bg-surface-dark border border-surface-border text-xs font-mono text-gray-300 hover:text-white hover:border-gray-500 flex items-center justify-center gap-1.5 transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Inspect All {feats.length} Features</span>
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </main>
  );
}
