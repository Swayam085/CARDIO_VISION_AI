/**
 * CardioVision AI — Clinical Input Form Component
 * ================================================
 * Production 55-feature clinical input form organized into 5 logical categories.
 * Strict leakage guard: LAD, LCX, RCA, Cath are strictly prohibited as form inputs.
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysis, CLINICAL_PRESETS } from '@/context/AnalysisContext';
import {
  User,
  Activity,
  HeartPulse,
  TestTube,
  Flame,
  CheckCircle2,
  Sparkles,
  ArrowRight,
  Loader2,
  ShieldCheck,
} from 'lucide-react';
import { PipelineErrorBanner } from '@/components/ui/PipelineErrorBanner';

// Multi-step loading indicator shows the current analysis stage
const STAGE_LABELS: Record<string, string> = {
  validating: 'Analyzing & Validating…',
  predicting: 'Running ML Models…',
  explaining: 'Generating Explanations…',
  preparing_3d: 'Preparing 3D Visualization…',
  complete:   'Complete!',
};

function LoadingIndicator({ stage }: { stage: string }) {
  return (
    <>
      <Loader2 className="w-4 h-4 animate-spin" />
      <span>{STAGE_LABELS[stage] ?? 'Running…'}</span>
    </>
  );
}

export function ClinicalForm() {
  const navigate = useNavigate();
  const {
    patientId,
    setPatientId,
    patientFeatures,
    setPatientFeatures,
    runAnalysis,
    applyPreset,
    activePresetKey,
    isLoading,
    loadingStage,
    error,
  } = useAnalysis();

  const [activeTab, setActiveTab] = useState<'demographics' | 'symptoms' | 'ecg' | 'labs' | 'echo'>('demographics');

  const handleInputChange = (field: string, value: any) => {
    setPatientFeatures((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await runAnalysis();
      navigate('/results');
    } catch {
      // Error state is captured and rendered via PipelineErrorBanner
    }
  };

  return (
    <form id="clinical-analysis-form" onSubmit={handleSubmit} className="space-y-6">
      {/* Sample Patient Presets Bar */}
      <div className="glass-card p-4 space-y-3 border-accent-500/20 bg-accent-500/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Sparkles className="w-4 h-4 text-accent-400" />
            <span>Sample Patient Clinical Presets</span>
          </div>
          <span className="text-xs text-gray-400 font-mono">1-Click Load</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {Object.entries(CLINICAL_PRESETS).map(([key, preset]) => (
            <button
              key={key}
              type="button"
              onClick={() => applyPreset(key)}
              className={`p-3 rounded-lg text-left transition-all border text-xs ${
                activePresetKey === key
                  ? 'bg-accent-500/20 border-accent-500 text-white shadow-lg'
                  : 'bg-surface-card/60 border-surface-border text-gray-300 hover:border-gray-500'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold">{preset.name}</span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${preset.badgeColor}`}>
                  {preset.badge}
                </span>
              </div>
              <p className="text-[11px] text-gray-400 line-clamp-2 leading-tight">{preset.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Patient ID Bar */}
      <div className="glass-card p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <User className="w-5 h-5 text-primary-400" />
          <div>
            <label htmlFor="patient-id-input" className="block text-xs font-semibold text-gray-300">
              Patient Identifier
            </label>
            <input
              id="patient-id-input"
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="bg-surface-dark border border-surface-border rounded px-2.5 py-1 text-sm text-white font-mono focus:border-primary-400 outline-none"
              placeholder="PT-10024"
            />
          </div>
        </div>

        {/* Leakage Guard Tag */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
          <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>Leakage Guard Active (LAD, LCX, RCA, Cath strictly excluded)</span>
        </div>
      </div>

      {/* Category Tabs Header */}
      <div className="flex border-b border-surface-border overflow-x-auto gap-1">
        {[
          { id: 'demographics', label: '1. Demographics & Vitals', icon: User },
          { id: 'symptoms', label: '2. Symptoms & Risk Factors', icon: Flame },
          { id: 'ecg', label: '3. ECG Findings', icon: Activity },
          { id: 'labs', label: '4. Laboratory Data', icon: TestTube },
          { id: 'echo', label: '5. Echocardiography', icon: HeartPulse },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-3 text-xs font-semibold font-display border-b-2 transition-all whitespace-nowrap ${
                isActive
                  ? 'border-primary-400 text-primary-400 bg-primary-500/5'
                  : 'border-transparent text-gray-400 hover:text-gray-200 hover:bg-surface-card/40'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-primary-400' : 'text-gray-500'}`} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab Panels */}
      <div className="glass-card p-6 min-h-[340px]">
        {/* TAB 1: Demographics & Vitals */}
        {activeTab === 'demographics' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 animate-fade-in">
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Age (Years)</label>
              <input
                type="number"
                min="18"
                max="100"
                value={patientFeatures.Age ?? 60}
                onChange={(e) => handleInputChange('Age', parseFloat(e.target.value) || 0)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Sex</label>
              <select
                value={patientFeatures.Sex ?? 'Male'}
                onChange={(e) => handleInputChange('Sex', e.target.value)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Weight (kg)</label>
              <input
                type="number"
                min="30"
                max="200"
                value={patientFeatures.Weight ?? 75}
                onChange={(e) => handleInputChange('Weight', parseFloat(e.target.value) || 0)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Length / Height (cm)</label>
              <input
                type="number"
                min="100"
                max="230"
                value={patientFeatures.Length ?? 170}
                onChange={(e) => handleInputChange('Length', parseFloat(e.target.value) || 0)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">BMI (kg/m²)</label>
              <input
                type="number"
                step="0.1"
                value={patientFeatures.BMI ?? 25.0}
                onChange={(e) => handleInputChange('BMI', parseFloat(e.target.value) || 0)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Systolic BP (mmHg)</label>
              <input
                type="number"
                min="70"
                max="240"
                value={patientFeatures.BP ?? 130}
                onChange={(e) => handleInputChange('BP', parseFloat(e.target.value) || 0)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Pulse Rate (bpm)</label>
              <input
                type="number"
                min="40"
                max="180"
                value={patientFeatures.PR ?? 75}
                onChange={(e) => handleInputChange('PR', parseFloat(e.target.value) || 0)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              />
            </div>
          </div>
        )}

        {/* TAB 2: Symptoms & Risk Factors */}
        {activeTab === 'symptoms' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 animate-fade-in">
            {[
              { key: 'DM', label: 'Diabetes Mellitus (DM)', type: 'binary' },
              { key: 'HTN', label: 'Hypertension (HTN)', type: 'binary' },
              { key: 'Current Smoker', label: 'Current Smoker', type: 'binary' },
              { key: 'EX-Smoker', label: 'Ex-Smoker', type: 'binary' },
              { key: 'FH', label: 'Family History (FH)', type: 'binary' },
              { key: 'Obesity', label: 'Obesity', type: 'yn' },
              { key: 'CRF', label: 'Chronic Renal Failure (CRF)', type: 'yn' },
              { key: 'CVA', label: 'Cerebrovascular Accident (CVA)', type: 'yn' },
              { key: 'Airway disease', label: 'Airway Disease', type: 'yn' },
              { key: 'Thyroid Disease', label: 'Thyroid Disease', type: 'yn' },
              { key: 'CHF', label: 'Congestive Heart Failure (CHF)', type: 'yn' },
              { key: 'DLP', label: 'Dyslipidemia (DLP)', type: 'yn' },
              { key: 'Edema', label: 'Peripheral Edema', type: 'binary' },
              { key: 'Weak Peripheral Pulse', label: 'Weak Peripheral Pulse', type: 'yn' },
              { key: 'Lung rales', label: 'Lung Rales', type: 'yn' },
              { key: 'Systolic Murmur', label: 'Systolic Murmur', type: 'yn' },
              { key: 'Diastolic Murmur', label: 'Diastolic Murmur', type: 'yn' },
              { key: 'Typical Chest Pain', label: 'Typical Chest Pain', type: 'binary' },
              { key: 'Dyspnea', label: 'Dyspnea / Shortness of Breath', type: 'yn' },
              { key: 'Atypical', label: 'Atypical Chest Pain', type: 'yn' },
              { key: 'Nonanginal', label: 'Non-anginal Pain', type: 'yn' },
              { key: 'Exertional CP', label: 'Exertional Chest Pain', type: 'yn' },
              { key: 'LowTH Ang', label: 'Low Threshold Angina', type: 'yn' },
            ].map((item) => (
              <div key={item.key}>
                <label className="block text-xs font-medium text-gray-300 mb-1">{item.label}</label>
                {item.type === 'binary' ? (
                  <select
                    value={patientFeatures[item.key] ?? 0}
                    onChange={(e) => handleInputChange(item.key, parseInt(e.target.value, 10))}
                    className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
                  >
                    <option value={1}>Yes (1)</option>
                    <option value={0}>No (0)</option>
                  </select>
                ) : (
                  <select
                    value={patientFeatures[item.key] ?? 'N'}
                    onChange={(e) => handleInputChange(item.key, e.target.value)}
                    className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
                  >
                    <option value="Y">Yes (Y)</option>
                    <option value="N">No (N)</option>
                  </select>
                )}
              </div>
            ))}
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">NYHA Function Class (1-4)</label>
              <select
                value={patientFeatures['Function Class'] ?? 1}
                onChange={(e) => handleInputChange('Function Class', parseInt(e.target.value, 10))}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              >
                <option value={1}>Class I (Normal)</option>
                <option value={2}>Class II (Mild)</option>
                <option value={3}>Class III (Moderate)</option>
                <option value={4}>Class IV (Severe)</option>
              </select>
            </div>
          </div>
        )}

        {/* TAB 3: ECG Findings */}
        {activeTab === 'ecg' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 animate-fade-in">
            {[
              { key: 'Q Wave', label: 'Q Wave', type: 'binary' },
              { key: 'St Elevation', label: 'ST Elevation', type: 'binary' },
              { key: 'St Depression', label: 'ST Depression', type: 'binary' },
              { key: 'Tinversion', label: 'T Wave Inversion', type: 'binary' },
              { key: 'LVH', label: 'Left Ventricular Hypertrophy (LVH)', type: 'yn' },
              { key: 'Poor R Progression', label: 'Poor R Wave Progression', type: 'yn' },
              { key: 'BBB', label: 'Bundle Branch Block (BBB)', type: 'yn' },
              { key: 'VHD', label: 'Valvular Heart Disease (VHD)', type: 'yn' },
            ].map((item) => (
              <div key={item.key}>
                <label className="block text-xs font-medium text-gray-300 mb-1">{item.label}</label>
                {item.type === 'binary' ? (
                  <select
                    value={patientFeatures[item.key] ?? 0}
                    onChange={(e) => handleInputChange(item.key, parseInt(e.target.value, 10))}
                    className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
                  >
                    <option value={1}>Present (1)</option>
                    <option value={0}>Absent (0)</option>
                  </select>
                ) : (
                  <select
                    value={patientFeatures[item.key] ?? 'N'}
                    onChange={(e) => handleInputChange(item.key, e.target.value)}
                    className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
                  >
                    <option value="Y">Present (Y)</option>
                    <option value="N">Absent (N)</option>
                  </select>
                )}
              </div>
            ))}
          </div>
        )}

        {/* TAB 4: Laboratory Data */}
        {activeTab === 'labs' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 animate-fade-in">
            {[
              { key: 'FBS', label: 'Fasting Blood Sugar (mg/dL)', defaultVal: 110 },
              { key: 'CR', label: 'Serum Creatinine (mg/dL)', defaultVal: 0.9, step: '0.1' },
              { key: 'TG', label: 'Triglycerides (mg/dL)', defaultVal: 150 },
              { key: 'LDL', label: 'LDL Cholesterol (mg/dL)', defaultVal: 120 },
              { key: 'HDL', label: 'HDL Cholesterol (mg/dL)', defaultVal: 45 },
              { key: 'BUN', label: 'Blood Urea Nitrogen (mg/dL)', defaultVal: 16 },
              { key: 'ESR', label: 'ESR (mm/hr)', defaultVal: 15 },
              { key: 'HB', label: 'Hemoglobin (g/dL)', defaultVal: 14.0, step: '0.1' },
              { key: 'K', label: 'Potassium (mEq/L)', defaultVal: 4.2, step: '0.1' },
              { key: 'Na', label: 'Sodium (mEq/L)', defaultVal: 140 },
              { key: 'WBC', label: 'WBC Count (cells/µL)', defaultVal: 7500 },
              { key: 'Lymph', label: 'Lymphocytes (%)', defaultVal: 30 },
              { key: 'Neut', label: 'Neutrophils (%)', defaultVal: 62 },
              { key: 'PLT', label: 'Platelets (x10³/µL)', defaultVal: 240 },
            ].map((item) => (
              <div key={item.key}>
                <label className="block text-xs font-medium text-gray-300 mb-1">{item.label}</label>
                <input
                  type="number"
                  step={item.step || '1'}
                  value={patientFeatures[item.key] ?? item.defaultVal}
                  onChange={(e) => handleInputChange(item.key, parseFloat(e.target.value) || 0)}
                  className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
                />
              </div>
            ))}
          </div>
        )}

        {/* TAB 5: Echocardiography */}
        {activeTab === 'echo' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 animate-fade-in max-w-xl">
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Ejection Fraction (EF-TTE %)</label>
              <input
                type="number"
                min="10"
                max="80"
                value={patientFeatures['EF-TTE'] ?? 55}
                onChange={(e) => handleInputChange('EF-TTE', parseFloat(e.target.value) || 0)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Region RWMA Score (0-3)</label>
              <select
                value={patientFeatures['Region RWMA'] ?? 0}
                onChange={(e) => handleInputChange('Region RWMA', parseInt(e.target.value, 10))}
                className="w-full bg-surface-dark border border-surface-border rounded-lg px-3 py-2 text-sm text-white focus:border-primary-400 outline-none"
              >
                <option value={0}>0 — Normal Wall Motion</option>
                <option value={1}>1 — Mild Regional Hypokinesia</option>
                <option value={2}>2 — Moderate Hypokinesia / Akinesia</option>
                <option value={3}>3 — Severe RWMA / Dyskinesia</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Error Banner */}
      {error && (
        <PipelineErrorBanner
          error={error}
          onRetry={() => runAnalysis().catch(() => {})}
        />
      )}

      {/* Action Footer */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
        <div className="text-xs text-gray-400 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>55 clinical variables prepared for model inference</span>
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full sm:w-auto px-8 py-3.5 rounded-xl bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-400 hover:to-accent-400 text-white font-display font-semibold text-sm shadow-lg shadow-primary-500/25 flex items-center justify-center gap-2 disabled:opacity-50 transition-all"
        >
          {isLoading ? (
            <LoadingIndicator stage={loadingStage} />
          ) : (
            <>
              <span>Run Multi-Vessel Risk Prediction</span>
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </form>
  );
}
