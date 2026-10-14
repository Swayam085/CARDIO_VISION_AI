/**
 * CardioVision AI — Core TypeScript Type Definitions
 * =====================================================
 * Shared types for frontend components, API contracts, and state management.
 */

// ── Vessel identifiers ────────────────────────────────────────
export type VesselId = 'CAD' | 'LAD' | 'LCX' | 'RCA';

export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical';

// ── Per-target prediction result ──────────────────────────────
export interface TaskPredictionResult {
  model_predicted_probability: number; // 0.0 to 1.0
  probability: number;                 // Alias for backward compatibility
  classification: string;              // 'CAD'/'Normal' or 'Stenotic'/'Normal'
  classification_binary: number;       // 1 or 0
  threshold: number;                   // Tuned threshold
  threshold_default: number;           // 0.5
  positive_class_label: string;        // 'CAD' or 'Stenotic'
  model_version: string;
}

// ── Overall predict API response ──────────────────────────────
export interface PredictResponse {
  patient_id?: string | null;
  predictions: Record<string, TaskPredictionResult>;
  model_version: string;
  terminology_notice: string;
  disclaimer: string;
}

// ── SHAP Feature attribution item ─────────────────────────────
export interface FeatureContribution {
  feature: string;
  patient_value: string | number | boolean;
  shap_value: number;
  absolute_contribution: number;
  direction: 'positive' | 'negative' | 'neutral';
  positive_negative_contribution: 'positive' | 'negative' | 'neutral';
  rank: number;
  feature_category?: 'Demographic' | 'Clinical' | 'Laboratory' | 'ECG' | 'Echo' | string;
}

// ── SHAP explain API response ─────────────────────────────────
export interface ExplainResponse {
  target: string;
  label: string;                        // "Model Feature Contribution"
  model_predicted_probability: number;
  base_value: number;
  top_positive_contributors: FeatureContribution[];
  top_negative_contributors: FeatureContribution[];
  top_contributors_overall: FeatureContribution[];
  all_contributions: FeatureContribution[];
  category_contributions?: Record<string, number>;
  disclaimer: string;
}

// ── Model Info API response ───────────────────────────────────
export interface ModelInfoResponse {
  model_version: string;
  model_types: Record<string, string>;
  models: Record<string, string>;
  training_dataset: string;
  dataset_source: string;
  dataset_size: { rows: number; columns: number };
  feature_count: number;
  validation_method: string;
  metrics: Record<string, any>;
  training_timestamp: string;
  thresholds: Record<string, number>;
  preprocessing_version: string;
  leakage_excluded_columns: string[];
  leakage_excluded: string[];
  safety_notice: string;
}

// ── Health API response ───────────────────────────────────────
export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error';
  timestamp: string;
  uptime_seconds: number;
  service: string;
  version: string;
  environment: string;
  python_version: string;
  platform: string;
  uci_dataset_url: string;
  anatomy_source_url: string;
}

// ── Metrics API response ──────────────────────────────────────
export interface TaskMetricSummary {
  selected_model: string;
  roc_auc: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_default: number;
  f1_tuned: number;
  threshold_default: number;
  threshold_optimized: number;
  cv_auc_mean?: number;
  cv_auc_std?: number;
  class_distribution?: Record<string, number>;
  confusion_matrix?: number[][];
}

export interface MetricsSummaryResponse {
  tasks: Record<string, TaskMetricSummary>;
  disclaimer: string;
  generated_at: string;
  dataset: {
    source?: string;
    total_records?: number;
    training_records?: number;
    test_records?: number;
    features?: number;
  };
}

export interface MetricsResponse {
  status: string;
  tasks?: Record<string, any>;
  [key: string]: any;
}

// ── 3D Anatomy Vessel Mesh Entry ──────────────────────────────
export interface VesselMesh {
  id: 'LAD' | 'LCX' | 'RCA';
  name: string;
  color: string;
  sourceAttribution: string;
}

export const VESSEL_MESH_REGISTRY: VesselMesh[] = [
  {
    id: 'LAD',
    name: 'Left Anterior Descending',
    color: '#ef4444',
    sourceAttribution: 'BodyParts3D / Anatomography by DBCLS (https://lifesciencedb.jp/bp3d)',
  },
  {
    id: 'LCX',
    name: 'Left Circumflex',
    color: '#f97316',
    sourceAttribution: 'BodyParts3D / Anatomography by DBCLS (https://lifesciencedb.jp/bp3d)',
  },
  {
    id: 'RCA',
    name: 'Right Coronary Artery',
    color: '#3b82f6',
    sourceAttribution: 'BodyParts3D / Anatomography by DBCLS (https://lifesciencedb.jp/bp3d)',
  },
];

// ── Navigation ────────────────────────────────────────────────
export interface NavRoute {
  path: string;
  label: string;
  description: string;
}

export const APP_ROUTES: NavRoute[] = [
  { path: '/',              label: 'Home',          description: 'CardioVision AI overview' },
  { path: '/analyze',       label: 'Analyze',       description: 'Enter patient clinical data' },
  { path: '/results',       label: 'Results',       description: 'View prediction results' },
  { path: '/visualization', label: 'Visualization', description: '3D vessel risk visualization' },
  { path: '/insights',      label: 'Insights',      description: 'SHAP feature importance' },
  { path: '/metrics',        label: 'Metrics',       description: 'Real model performance metrics' },
  { path: '/report',        label: 'Report',        description: 'Clinical summary report' },
  { path: '/documentation', label: 'Documentation', description: 'Dataset & model documentation' },
];

// ── Risk Helpers ──────────────────────────────────────────────
export function getRiskLevel(probability: number, threshold: number = 0.5): RiskLevel {
  if (probability < threshold * 0.6) return 'low';
  if (probability < threshold)       return 'moderate';
  if (probability < threshold * 1.3) return 'high';
  return 'critical';
}

export function getRiskColor(level: RiskLevel): string {
  const map: Record<RiskLevel, string> = {
    low:      '#22c55e',
    moderate: '#eab308',
    high:     '#f97316',
    critical: '#ef4444',
  };
  return map[level];
}
