/**
 * CardioVision AI — API Service Layer
 * =====================================
 * All backend communication goes through this module.
 * Configured via VITE_API_BASE_URL environment variable.
 */

import axios, { type AxiosInstance, type AxiosError } from 'axios';
import type {
  HealthResponse,
  ModelInfoResponse,
  MetricsResponse,
  PredictResponse,
  ExplainResponse,
} from '@/types';

// ── Unified Analyze response (predict + SHAP combined) ────────
export interface AnalyzeResponse {
  patient_id?: string | null;
  predictions: PredictResponse['predictions'];
  model_version: string;
  terminology_notice: string;
  disclaimer: string;
  shap_explanations?: Record<string, ExplainResponse>;
}

const resolveApiBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (!envUrl) return 'http://localhost:8000';
  if (envUrl.startsWith('http://') || envUrl.startsWith('https://')) {
    return envUrl;
  }
  return `https://${envUrl}`;
};

// ── Axios instance ────────────────────────────────────────────
const api: AxiosInstance = axios.create({
  baseURL: resolveApiBaseUrl(),
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Request interceptor ───────────────────────────────────────
api.interceptors.request.use(
  (config) => {
    if (import.meta.env.DEV) {
      console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// ── Response interceptor ──────────────────────────────────────
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const status = error.response?.status;
    const data   = error.response?.data as Record<string, unknown> | undefined;

    if (status === 503) {
      console.info('[API] ML pipeline not yet ready (503). Train the model first.');
    } else if (status === 422) {
      console.warn('[API] Unprocessable Entity / Data Leakage / Input Error:', data);
    } else if (status === 500) {
      console.error('[API] Internal server error:', data);
    }

    return Promise.reject(error);
  },
);

// ── Health ────────────────────────────────────────────────────
export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}

// ── Model Info ────────────────────────────────────────────────
export async function getModelInfo(): Promise<ModelInfoResponse> {
  const { data } = await api.get<ModelInfoResponse>('/api/v1/model/info');
  return data;
}

// ── Metrics ───────────────────────────────────────────────────
export async function getMetrics(): Promise<MetricsResponse> {
  const { data } = await api.get<MetricsResponse>('/api/v1/metrics');
  return data;
}

export async function getMetricsSummary(): Promise<any> {
  const { data } = await api.get('/api/v1/metrics/summary');
  return data;
}

// ── Predict ───────────────────────────────────────────────────
export async function predictApi(
  features: Record<string, unknown>,
  patientId?: string
): Promise<PredictResponse> {
  const payload = {
    patient_id: patientId || undefined,
    features,
  };
  const { data } = await api.post<PredictResponse>('/api/v1/predict', payload);
  return data;
}

// ── Explain (SHAP) ────────────────────────────────────────────
export async function explainApi(
  features: Record<string, unknown>,
  target: string = 'CAD',
  topN: number = 10
): Promise<ExplainResponse> {
  const payload = {
    features,
    target: target.toUpperCase(),
    top_n: topN,
  };
  const { data } = await api.post<ExplainResponse>('/api/v1/explain', payload);
  return data;
}

// ── Unified Analyze (predict + SHAP in one shot) ──────────────
export async function analyzeApi(
  features: Record<string, unknown>,
  patientId?: string,
  topN: number = 10,
): Promise<AnalyzeResponse> {
  const payload = {
    patient_id: patientId || undefined,
    features,
    top_n: topN,
  };
  const { data } = await api.post<AnalyzeResponse>('/api/v1/analyze', payload);
  return data;
}

export default api;
