/**
 * CardioVision AI — Visualization Scale Legend Component
 * ========================================================
 * Complies with Requirement 8:
 * Clearly labeled visualization scale (0-30, 30-60, 60-80, 80-100).
 * Prominently displays: "Visualization scale — not a clinical severity classification."
 * Explicitly states that values are model-predicted probabilities, NOT percentage blockage.
 */

import { VISUALIZATION_SCALE_TIERS } from '@/services/vesselRegistry';
import { Info } from 'lucide-react';

interface Props {
  className?: string;
  compact?: boolean;
}

export function VisualizationScaleLegend({ className = '', compact = false }: Props) {
  return (
    <div
      id="visualization-scale-legend"
      className={`rounded-xl border border-surface-border bg-surface-dark/80 backdrop-blur p-4 text-xs font-mono ${className}`}
    >
      <div className="flex items-center justify-between gap-2 pb-2.5 mb-2.5 border-b border-surface-border/70">
        <div className="flex items-center gap-2 text-gray-200 font-semibold">
          <span className="w-2 h-2 rounded-full bg-accent-400 animate-pulse" />
          <span>Vessel Risk Color Scale</span>
        </div>
        <span className="text-[10px] uppercase tracking-wider text-amber-400/90 font-bold px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20">
          Non-Clinical Scale
        </span>
      </div>

      {/* Mandatory Disclaimer Label */}
      <div className="flex items-start gap-2 p-2.5 rounded-lg bg-surface-darker/60 border border-surface-border/50 text-[11px] text-gray-300 mb-3 leading-relaxed">
        <Info className="w-3.5 h-3.5 text-accent-400 shrink-0 mt-0.5" aria-hidden="true" />
        <div>
          <strong className="text-white block font-medium">
            Visualization scale — not a clinical severity classification.
          </strong>
          <span className="text-gray-400">
            Colors reflect model-predicted probability (0–100%), <strong>not anatomical percentage blockage</strong>.
          </span>
        </div>
      </div>

      {/* Tier Badges / Grid */}
      <div className={`grid ${compact ? 'grid-cols-2 gap-2' : 'grid-cols-1 sm:grid-cols-4 gap-2.5'}`}>
        {VISUALIZATION_SCALE_TIERS.map((tier) => (
          <div
            key={tier.range}
            className={`p-2.5 rounded-lg border flex flex-col justify-between transition-all ${tier.bgClass} ${tier.borderClass}`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-bold text-xs" style={{ color: tier.color }}>
                {tier.range}
              </span>
              <span
                className="w-2.5 h-2.5 rounded-full shadow-sm"
                style={{ backgroundColor: tier.color, boxShadow: `0 0 8px ${tier.color}80` }}
              />
            </div>
            <div className="font-medium text-[11px] text-white">{tier.label}</div>
            {!compact && (
              <p className="text-[10px] text-gray-400 mt-1 leading-snug line-clamp-2">
                {tier.description}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
