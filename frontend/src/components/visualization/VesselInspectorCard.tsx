/**
 * CardioVision AI — Vessel Inspector Card Component
 * ===================================================
 * Complies with Requirement 9:
 * Displays selected vessel detailed inspection:
 *   - Vessel name (Left Anterior Descending, Left Circumflex, Right Coronary Artery)
 *   - Model-predicted probability from POST /analyze
 *   - Tuned classification
 *   - Top model contributing features (SHAP)
 *   - Anatomical territory and BodyParts3D / FMA provenance
 */

import type { TaskPredictionResult, ExplainResponse } from '@/types';
import {
  type CoronaryVesselId,
  getVesselConfig,
  getRiskScaleTier,
} from '@/services/vesselRegistry';
import {
  Layers,
  Sparkles,
  TrendingUp,
  TrendingDown,
  Loader2,
} from 'lucide-react';

interface Props {
  selectedVessel: CoronaryVesselId;
  onSelectVessel: (id: CoronaryVesselId) => void;
  prediction?: TaskPredictionResult;
  shapExplanation?: ExplainResponse | null;
  isLoadingShap?: boolean;
}

export function VesselInspectorCard({
  selectedVessel,
  onSelectVessel,
  prediction,
  shapExplanation,
  isLoadingShap = false,
}: Props) {
  const config = getVesselConfig(selectedVessel);

  // Real probability from API or default placeholder if not analyzed yet
  const proba = prediction ? prediction.model_predicted_probability : 0.25;
  const thresh = prediction ? prediction.threshold : 0.5;
  const isStenotic = prediction
    ? prediction.classification_binary === 1 || proba >= thresh
    : proba >= thresh;

  const tier = getRiskScaleTier(proba);
  const probPct = (proba * 100).toFixed(1);

  // Top SHAP contributors
  const topFeatures = shapExplanation?.top_contributors_overall?.slice(0, 5) ?? [];

  return (
    <div
      id="vessel-inspector-card"
      className="glass-card p-5 lg:p-6 flex flex-col justify-between border-surface-border space-y-6"
    >
      {/* Top Header & Selector */}
      <div>
        <div className="flex items-center justify-between border-b border-surface-border pb-3 mb-4">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: tier.color }} />
            <span className="text-xs font-mono font-bold tracking-wider text-gray-300 uppercase">
              Vessel Inspector
            </span>
          </div>
          <span
            className={`px-2.5 py-0.5 rounded text-xs font-bold font-mono uppercase border ${tier.bgClass} ${tier.borderClass} ${tier.textClass}`}
          >
            {tier.label}
          </span>
        </div>

        {/* Vessel Title */}
        <div className="mb-4">
          <div className="flex items-baseline gap-2">
            <h2 className="text-2xl font-display font-extrabold text-white tracking-tight">
              {config.id}
            </h2>
            <span className="text-sm font-semibold text-gray-300">
              — {config.fullName}
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1 leading-relaxed">
            {config.clinicalName}
          </p>
        </div>

        {/* Prediction Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          {/* Probability Box */}
          <div className="p-3.5 rounded-xl bg-surface-dark/80 border border-surface-border">
            <span className="text-[10px] font-mono uppercase text-gray-400 tracking-wider block mb-1">
              Model Probability
            </span>
            <div className="flex items-baseline gap-1.5">
              <span className="text-2xl font-display font-black text-white">
                {probPct}%
              </span>
              <span className="text-[11px] font-mono text-gray-500">
                ({proba.toFixed(4)})
              </span>
            </div>
            <div className="w-full bg-surface-darker h-1.5 rounded-full mt-2 overflow-hidden border border-surface-border/50">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, proba * 100)}%`, backgroundColor: tier.color }}
              />
            </div>
          </div>

          {/* Classification Box */}
          <div className="p-3.5 rounded-xl bg-surface-dark/80 border border-surface-border flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-mono uppercase text-gray-400 tracking-wider block mb-1">
                Classification
              </span>
              <span
                className="text-sm font-display font-bold block"
                style={{ color: tier.color }}
              >
                {isStenotic ? 'Stenotic (High Risk)' : 'Normal (Low Risk)'}
              </span>
            </div>
            <div className="text-[11px] font-mono text-gray-400 mt-2">
              Decision Threshold: <strong className="text-gray-200">{thresh.toFixed(2)}</strong>
            </div>
          </div>
        </div>

        {/* Anatomical Territory & FMA Details */}
        <div className="p-3 rounded-lg bg-surface-darker/70 border border-surface-border/60 text-xs font-mono space-y-1.5 mb-5">
          <div className="flex items-start gap-1.5 text-gray-300">
            <Layers className="w-3.5 h-3.5 text-accent-400 shrink-0 mt-0.5" />
            <div>
              <span className="text-gray-400">Territory: </span>
              <span className="text-gray-200">{config.anatomicalTerritory}</span>
            </div>
          </div>
          <div className="flex items-center gap-2 pt-1 border-t border-surface-border/40 text-[11px] text-gray-400">
            <span>FMA ID: <code className="text-primary-300">{config.fmaId}</code></span>
            <span>·</span>
            <span>BP Mesh: <code className="text-emerald-300">{config.meshName}</code></span>
          </div>
        </div>

        {/* Top Contributing Features (SHAP) */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-gray-300 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>Top Model Contributing Features</span>
            </span>
            {isLoadingShap && (
              <span className="text-[10px] font-mono text-gray-400 flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin text-accent-400" />
                <span>Computing SHAP…</span>
              </span>
            )}
          </div>

          {topFeatures.length > 0 ? (
            <div className="space-y-2">
              {topFeatures.map((feat, idx) => {
                const isPositive = feat.direction === 'positive' || feat.shap_value > 0;
                return (
                  <div
                    key={`${feat.feature}-${idx}`}
                    className="p-2 rounded-lg bg-surface-dark/70 border border-surface-border/60 flex items-center justify-between text-xs font-mono"
                  >
                    <div className="flex items-center gap-2 min-w-0 pr-2">
                      {isPositive ? (
                        <TrendingUp className="w-3.5 h-3.5 text-rose-400 shrink-0" aria-label="Increases risk" />
                      ) : (
                        <TrendingDown className="w-3.5 h-3.5 text-emerald-400 shrink-0" aria-label="Decreases risk" />
                      )}
                      <div className="truncate">
                        <span className="text-gray-200 font-medium block truncate">{feat.feature}</span>
                        <span className="text-[10px] text-gray-400">Value: {String(feat.patient_value)}</span>
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <span
                        className={`font-mono text-xs font-bold ${
                          isPositive ? 'text-rose-400' : 'text-emerald-400'
                        }`}
                      >
                        {isPositive ? '+' : ''}{feat.shap_value.toFixed(3)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="p-3 rounded-lg bg-surface-dark/40 border border-surface-border/40 text-center text-xs text-gray-400 font-mono">
              {isLoadingShap
                ? 'Extracting TreeSHAP feature attributions for this vessel…'
                : 'Run analysis or select a clinical preset to view SHAP feature attributions.'}
            </div>
          )}
        </div>
      </div>

      {/* Vessel Quick-Switch Tabs */}
      <div className="pt-4 border-t border-surface-border space-y-2">
        <span className="text-[11px] font-mono text-gray-400 block font-semibold">
          Select Coronary Vessel to Inspect & Focus Camera:
        </span>
        <div className="grid grid-cols-3 gap-2">
          {(['LAD', 'LCX', 'RCA'] as const).map((vId) => {
            const isSelected = selectedVessel === vId;
            return (
              <button
                key={vId}
                id={`btn-select-vessel-${vId.toLowerCase()}`}
                onClick={() => onSelectVessel(vId)}
                className={`py-2 px-3 rounded-lg text-xs font-mono font-bold transition-all border flex flex-col items-center gap-0.5 ${
                  isSelected
                    ? 'bg-primary-500/20 border-primary-500 text-white shadow-lg shadow-primary-500/10 ring-1 ring-primary-500/50'
                    : 'bg-surface-dark border-surface-border text-gray-400 hover:text-white hover:bg-surface-dark/90'
                }`}
              >
                <span>{vId}</span>
                <span className="text-[9px] font-normal opacity-70">
                  {vId === 'LAD' ? 'Anterior' : vId === 'LCX' ? 'Lateral' : 'Right'}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
