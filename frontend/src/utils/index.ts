/**
 * CardioVision AI — Utility Functions
 */

/**
 * Format a risk score (0.0–1.0) as percentage string.
 */
export function formatRiskPercent(score: number): string {
  return `${(score * 100).toFixed(1)}%`;
}

/**
 * Clamp a value between min and max.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Format ISO timestamp to locale string.
 */
export function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

/**
 * Map 0–1 risk score to a Tailwind color class.
 */
export function riskScoreToColorClass(score: number): string {
  if (score < 0.25) return 'text-green-400';
  if (score < 0.5)  return 'text-yellow-400';
  if (score < 0.75) return 'text-orange-400';
  return 'text-red-400';
}

/**
 * Type-safe environment variable getter.
 */
export function getEnv(key: string, fallback = ''): string {
  return (import.meta.env[key] as string | undefined) ?? fallback;
}

// ── Feature Categories (Exact UCI ID 411 Dataset Mapping) ─────
export type FeatureCategory = 'Demographic' | 'Clinical' | 'Laboratory' | 'ECG' | 'Echo';

export const DEMOGRAPHIC_FEATURES = new Set([
  'Age', 'Weight', 'Length', 'Sex', 'BMI',
]);

export const ECG_FEATURES = new Set([
  'Q Wave', 'St Elevation', 'St Depression', 'Tinversion', 'LVH', 'Poor R Progression', 'BBB', 'VHD',
]);

export const LAB_FEATURES = new Set([
  'FBS', 'CR', 'TG', 'LDL', 'HDL', 'BUN', 'ESR', 'HB', 'K', 'Na', 'WBC', 'Lymph', 'Neut', 'PLT',
]);

export const ECHO_FEATURES = new Set([
  'EF-TTE', 'Region RWMA',
]);

export function getFeatureCategory(featureName: string): FeatureCategory {
  if (DEMOGRAPHIC_FEATURES.has(featureName)) return 'Demographic';
  if (ECHO_FEATURES.has(featureName)) return 'Echo';
  if (ECG_FEATURES.has(featureName)) return 'ECG';
  if (LAB_FEATURES.has(featureName)) return 'Laboratory';
  return 'Clinical';
}

export const CATEGORY_STYLES: Record<FeatureCategory, { badge: string; text: string; bg: string; border: string }> = {
  Demographic: {
    badge: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
    text: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/30',
  },
  Clinical: {
    badge: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
    text: 'text-purple-400',
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/30',
  },
  Laboratory: {
    badge: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
    text: 'text-cyan-400',
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/30',
  },
  ECG: {
    badge: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
    text: 'text-yellow-400',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
  },
  Echo: {
    badge: 'bg-rose-500/15 text-rose-400 border-rose-500/30',
    text: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/30',
  },
};

export const MODEL_CONTRIBUTION_LABEL = "Feature contribution to model output";
export const NON_CAUSAL_DISCLAIMER =
  "Feature contribution to model output (TreeSHAP) quantifies statistical feature attributions to the algorithmic output. These values represent mathematical contributions to the model's prediction score and do not imply biological causality, etiology, or direct mechanism. Never interpret feature contributions as 'this feature caused CAD'.";

