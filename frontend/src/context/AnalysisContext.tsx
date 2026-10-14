/**
 * CardioVision AI — Analysis Context & Global State Store
 * ========================================================
 * Persists patient feature inputs, prediction results, and SHAP explanations
 * across all frontend routes (/analyze, /results, /visualization, /insights, /report).
 *
 * Calls the unified /api/v1/analyze endpoint (predict + SHAP in one shot),
 * with fallback to /predict + /explain if analyze is unavailable.
 */

import React, { createContext, useContext, useState, type ReactNode } from 'react';
import type { PredictResponse, ExplainResponse } from '@/types';
import { predictApi, explainApi, analyzeApi } from '@/services/api';

export type LoadingStage =
  | 'idle'
  | 'validating'
  | 'predicting'
  | 'explaining'
  | 'preparing_3d'
  | 'complete';

export interface ClinicalPreset {
  id: string;
  name: string;
  description: string;
  badge: string;
  badgeColor: string;
  features: Record<string, any>;
}

export const CLINICAL_PRESETS: Record<string, ClinicalPreset> = {
  high_risk: {
    id: 'high_risk',
    name: 'High Risk Multi-Vessel CAD',
    description: '68yo male with Diabetes, HTN, Smoker, EF 40%, ST depression, Q wave & RWMA.',
    badge: 'High Risk',
    badgeColor: 'bg-red-500/15 text-red-400 border-red-500/30',
    features: {
      Age: 68, Weight: 85, Length: 170, Sex: 'Male', BMI: 29.4,
      DM: 1, HTN: 1, 'Current Smoker': 1, 'EX-Smoker': 0, FH: 1,
      Obesity: 'Y', CRF: 'N', CVA: 'N', 'Airway disease': 'N', 'Thyroid Disease': 'N',
      CHF: 'Y', DLP: 'Y', BP: 155, PR: 88, Edema: 1,
      'Weak Peripheral Pulse': 'Y', 'Lung rales': 'Y', 'Systolic Murmur': 'N', 'Diastolic Murmur': 'N',
      'Typical Chest Pain': 1, Dyspnea: 'Y', 'Function Class': 3,
      Atypical: 'N', Nonanginal: 'N', 'Exertional CP': 'Y', 'LowTH Ang': 'Y',
      'Q Wave': 1, 'St Elevation': 0, 'St Depression': 1, Tinversion: 1,
      LVH: 'Y', 'Poor R Progression': 'Y', BBB: 'N', VHD: 'N',
      FBS: 165, CR: 1.3, TG: 240, LDL: 160, HDL: 35, BUN: 22, ESR: 28,
      HB: 13.2, K: 4.4, Na: 138, WBC: 9200, Lymph: 24, Neut: 70, PLT: 280,
      'EF-TTE': 40, 'Region RWMA': 2,
    },
  },
  moderate_risk: {
    id: 'moderate_risk',
    name: 'Moderate / Single Vessel Risk',
    description: '58yo male with HTN, DLP, Exertional Chest Pain, ST Depression.',
    badge: 'Moderate Risk',
    badgeColor: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
    features: {
      Age: 58, Weight: 76, Length: 174, Sex: 'Male', BMI: 25.1,
      DM: 0, HTN: 1, 'Current Smoker': 0, 'EX-Smoker': 1, FH: 1,
      Obesity: 'N', CRF: 'N', CVA: 'N', 'Airway disease': 'N', 'Thyroid Disease': 'N',
      CHF: 'N', DLP: 'Y', BP: 138, PR: 74, Edema: 0,
      'Weak Peripheral Pulse': 'N', 'Lung rales': 'N', 'Systolic Murmur': 'N', 'Diastolic Murmur': 'N',
      'Typical Chest Pain': 1, Dyspnea: 'N', 'Function Class': 2,
      Atypical: 'N', Nonanginal: 'N', 'Exertional CP': 'Y', 'LowTH Ang': 'N',
      'Q Wave': 0, 'St Elevation': 0, 'St Depression': 1, Tinversion: 1,
      LVH: 'N', 'Poor R Progression': 'N', BBB: 'N', VHD: 'N',
      FBS: 108, CR: 0.9, TG: 175, LDL: 132, HDL: 44, BUN: 16, ESR: 14,
      HB: 14.5, K: 4.2, Na: 140, WBC: 7400, Lymph: 30, Neut: 62, PLT: 240,
      'EF-TTE': 52, 'Region RWMA': 1,
    },
  },
  low_risk: {
    id: 'low_risk',
    name: 'Low Risk / Normal Features',
    description: '42yo female, non-smoker, normal BP, clear ECG, EF 62%, no chest pain.',
    badge: 'Low Risk',
    badgeColor: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
    features: {
      Age: 42, Weight: 62, Length: 168, Sex: 'Female', BMI: 22.0,
      DM: 0, HTN: 0, 'Current Smoker': 0, 'EX-Smoker': 0, FH: 0,
      Obesity: 'N', CRF: 'N', CVA: 'N', 'Airway disease': 'N', 'Thyroid Disease': 'N',
      CHF: 'N', DLP: 'N', BP: 118, PR: 68, Edema: 0,
      'Weak Peripheral Pulse': 'N', 'Lung rales': 'N', 'Systolic Murmur': 'N', 'Diastolic Murmur': 'N',
      'Typical Chest Pain': 0, Dyspnea: 'N', 'Function Class': 1,
      Atypical: 'N', Nonanginal: 'N', 'Exertional CP': 'N', 'LowTH Ang': 'N',
      'Q Wave': 0, 'St Elevation': 0, 'St Depression': 0, Tinversion: 0,
      LVH: 'N', 'Poor R Progression': 'N', BBB: 'N', VHD: 'N',
      FBS: 88, CR: 0.8, TG: 110, LDL: 95, HDL: 58, BUN: 14, ESR: 8,
      HB: 14.0, K: 4.1, Na: 141, WBC: 6200, Lymph: 35, Neut: 58, PLT: 220,
      'EF-TTE': 62, 'Region RWMA': 0,
    },
  },
};

interface AnalysisContextType {
  patientId: string;
  setPatientId: (id: string) => void;
  patientFeatures: Record<string, any>;
  setPatientFeatures: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  predictionResult: PredictResponse | null;
  explainResult: ExplainResponse | null;
  explainResults: Record<string, ExplainResponse>;
  activeExplainTarget: string;
  isLoading: boolean;
  loadingStage: LoadingStage;
  error: string | null;
  activePresetKey: string | null;
  runAnalysis: (featuresOverride?: Record<string, any>, patientIdOverride?: string) => Promise<PredictResponse>;
  fetchExplanation: (target: string) => Promise<ExplainResponse>;
  applyPreset: (presetKey: string) => void;
  resetAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [patientId, setPatientId] = useState<string>('PT-10024');
  const [patientFeatures, setPatientFeatures] = useState<Record<string, any>>(CLINICAL_PRESETS.high_risk.features);
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);
  const [explainResult, setExplainResult] = useState<ExplainResponse | null>(null);
  const [explainResults, setExplainResults] = useState<Record<string, ExplainResponse>>({});
  const [activeExplainTarget, setActiveExplainTarget] = useState<string>('CAD');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingStage, setLoadingStage] = useState<LoadingStage>('idle');
  const [error, setError] = useState<string | null>(null);
  const [activePresetKey, setActivePresetKey] = useState<string | null>('high_risk');

  const applyPreset = (presetKey: string) => {
    const preset = CLINICAL_PRESETS[presetKey];
    if (preset) {
      setPatientFeatures(preset.features);
      setActivePresetKey(presetKey);
      setPredictionResult(null);
      setExplainResult(null);
      setExplainResults({});
    }
  };

  const runAnalysis = async (featuresOverride?: Record<string, any>, patientIdOverride?: string): Promise<PredictResponse> => {
    setIsLoading(true);
    setError(null);
    setLoadingStage('validating');

    const targetFeatures = featuresOverride || patientFeatures;
    const targetPid = patientIdOverride || patientId;

    try {
      // Try the unified /analyze endpoint first (prediction + SHAP combined)
      setLoadingStage('predicting');

      try {
        const analyzeRes = await analyzeApi(targetFeatures, targetPid, 10);

        // Build a compatible PredictResponse from analyze result
        const predRes: PredictResponse = {
          patient_id: analyzeRes.patient_id,
          predictions: analyzeRes.predictions,
          model_version: analyzeRes.model_version,
          terminology_notice: analyzeRes.terminology_notice,
          disclaimer: analyzeRes.disclaimer,
        };
        setPredictionResult(predRes);

        // Extract CAD SHAP explanation if available
        setLoadingStage('explaining');
        if (analyzeRes.shap_explanations?.CAD) {
          setExplainResult(analyzeRes.shap_explanations.CAD);
          setExplainResults({ CAD: analyzeRes.shap_explanations.CAD });
          setActiveExplainTarget('CAD');
        } else {
          // Fallback: fetch SHAP separately
          try {
            const expRes = await explainApi(targetFeatures, 'CAD', 10);
            setExplainResult(expRes);
            setExplainResults({ CAD: expRes });
            setActiveExplainTarget('CAD');
          } catch (shapErr) {
            console.warn('SHAP explanation warning (analyze path):', shapErr);
          }
        }

        setLoadingStage('preparing_3d');
        setLoadingStage('complete');
        setIsLoading(false);
        return predRes;

      } catch (analyzeErr: any) {
        // If 422 or 503, immediately rethrow with structured detail
        const status = analyzeErr.response?.status;
        if (status === 422 || status === 503) {
          throw analyzeErr;
        }

        // Fallback to separate predict + explain only on connection/network retry
        console.warn('[API] Unified /analyze failed, attempting fallback:', analyzeErr);
        setLoadingStage('predicting');

        const predRes = await predictApi(targetFeatures, targetPid);
        setPredictionResult(predRes);

        setLoadingStage('explaining');
        try {
          const expRes = await explainApi(targetFeatures, 'CAD', 10);
          setExplainResult(expRes);
          setExplainResults({ CAD: expRes });
          setActiveExplainTarget('CAD');
        } catch (shapErr) {
          console.warn('Initial SHAP explanation warning:', shapErr);
        }

        setLoadingStage('preparing_3d');
        setLoadingStage('complete');
        setIsLoading(false);
        return predRes;
      }
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const msg = typeof detail === 'string'
        ? detail
        : detail?.message
        ? detail.message
        : Array.isArray(detail)
        ? detail.map((d: any) => `${d.loc?.slice(-1)[0] || 'field'}: ${d.msg}`).join(', ')
        : err.message || 'Analysis prediction failed';
      setError(msg);
      setLoadingStage('idle');
      setIsLoading(false);
      throw err;
    }
  };

  const fetchExplanation = async (target: string): Promise<ExplainResponse> => {
    const normTarget = target.toUpperCase();
    if (explainResults[normTarget]) {
      setExplainResult(explainResults[normTarget]);
      setActiveExplainTarget(normTarget);
      return explainResults[normTarget];
    }

    setIsLoading(true);
    setLoadingStage('explaining');
    try {
      const expRes = await explainApi(patientFeatures, normTarget, 10);
      setExplainResult(expRes);
      setExplainResults((prev) => ({ ...prev, [normTarget]: expRes }));
      setActiveExplainTarget(normTarget);
      setLoadingStage('complete');
      setIsLoading(false);
      return expRes;
    } catch (err: any) {
      console.error('Explanation error:', err);
      setLoadingStage('idle');
      setIsLoading(false);
      throw err;
    }
  };

  const resetAnalysis = () => {
    setPredictionResult(null);
    setExplainResult(null);
    setExplainResults({});
    setError(null);
    setLoadingStage('idle');
  };

  return (
    <AnalysisContext.Provider
      value={{
        patientId,
        setPatientId,
        patientFeatures,
        setPatientFeatures,
        predictionResult,
        explainResult,
        explainResults,
        activeExplainTarget,
        isLoading,
        loadingStage,
        error,
        activePresetKey,
        runAnalysis,
        fetchExplanation,
        applyPreset,
        resetAnalysis,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
}
