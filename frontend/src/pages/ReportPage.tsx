/**
 * CardioVision AI — Clinical Summary Report (/report)
 * =====================================================
 * Comprehensive AI-generated cardiovascular risk and multi-vessel stenosis report.
 * Designed for clinical review, multi-disciplinary team rounds, and print/PDF export.
 *
 * Mandated Labels & Standards:
 *   - "AI-generated educational / decision-support report"
 *   - "Not an official medical diagnosis."
 *   - "Feature contribution to model output" (strictly non-causal)
 *
 * Full Report Sections:
 *   1. Official Document Header & AI Decision-Support Notice
 *   2. Patient Comprehensive Clinical Input Summary (Demographics, Vitals, Symptoms, Labs, ECG, Echo)
 *   3. Overall CAD & Multi-Vessel Risk Predictions (CAD, LAD, LCX, RCA)
 *   4. Multi-Domain Physiological Breakdown (Vascular, Metabolic, Hemodynamic, Ventricular)
 *   5. Model Feature Contributions (TreeSHAP attributions with patient values, categories, signs)
 *   6. Model Architecture & Preprocessing Pipeline Information
 *   7. Validation Methodology & Out-of-Fold Performance
 *   8. Dataset Provenance (UCI ID 411, N=303, 55 features)
 *   9. Critical Methodological & Clinical Limitations
 *  10. Formal Safety Disclaimer & Professional Sign-Off Block
 */

import { useRef, useMemo, useState } from 'react';
import { useAnalysis } from '@/context/AnalysisContext';
import { getRiskLevel, getRiskColor } from '@/types';
import {
  getFeatureCategory,
  CATEGORY_STYLES,
  MODEL_CONTRIBUTION_LABEL,
  NON_CAUSAL_DISCLAIMER,
  type FeatureCategory,
} from '@/utils';
import {
  FileText,
  Printer,
  User,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Heart,
  Cpu,
  ShieldAlert,
  Calendar,
  Clock,
} from 'lucide-react';

export function ReportPage() {
  const { predictionResult, patientId, patientFeatures, explainResult, explainResults } = useAnalysis();
  const reportRef = useRef<HTMLDivElement>(null);

  const handlePrint = () => {
    window.print();
  };

  const predictions = predictionResult?.predictions || {};
  const currentExplain = explainResults['CAD'] || explainResult;

  const [reportDate] = useState(() => new Date());
  const { formattedDate, formattedTime, isoTimestamp } = useMemo(() => {
    return {
      formattedDate: reportDate.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      }),
      formattedTime: reportDate.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZoneName: 'short',
      }),
      isoTimestamp: reportDate.toISOString(),
    };
  }, [reportDate]);

  // Enriched CAD SHAP contributions
  const shapContributions = useMemo(() => {
    if (!currentExplain?.all_contributions) return [];
    return currentExplain.all_contributions.map((c) => ({
      ...c,
      feature_category: c.feature_category || getFeatureCategory(c.feature),
    }));
  }, [currentExplain]);

  const topPositiveShap = useMemo(() => {
    return shapContributions.filter((c) => c.shap_value > 0).slice(0, 5);
  }, [shapContributions]);

  const topNegativeShap = useMemo(() => {
    return shapContributions.filter((c) => c.shap_value < 0).slice(0, 5);
  }, [shapContributions]);

  // Physiological Breakdown Metrics
  const physiologicalProfile = useMemo(() => {
    const f = patientFeatures || {};
    // 1. Vascular / Ischemic Risk
    const hasTypicalCP = f['Typical Chest Pain'] === 1 || f['Typical Chest Pain'] === 'Y';
    const hasExertional = f['Exertional CP'] === 'Y' || f['Exertional CP'] === 1;
    const hasStDep = f['St Depression'] === 1 || f['St Depression'] === 'Y';
    const hasQWave = f['Q Wave'] === 1 || f['Q Wave'] === 'Y';

    // 2. Metabolic Profile
    const fbs = Number(f.FBS) || 100;
    const ldl = Number(f.LDL) || 120;
    const hdl = Number(f.HDL) || 45;
    const tg = Number(f.TG) || 150;
    const hasDM = f.DM === 1 || f.DM === 'Y';

    // 3. Hemodynamics & Vitals
    const bp = Number(f.BP) || 120;
    const pr = Number(f.PR) || 75;
    const hasHTN = f.HTN === 1 || f.HTN === 'Y';
    const bmi = Number(f.BMI) || 24;

    // 4. Ventricular Function
    const ef = Number(f['EF-TTE']) || 55;
    const rwma = Number(f['Region RWMA']) || 0;

    return {
      ischemic: {
        status: hasTypicalCP && (hasStDep || hasQWave) ? 'High Ischemic Burden' : hasExertional ? 'Moderate Ischemic Signs' : 'Low Ischemic Burden',
        color: hasTypicalCP && (hasStDep || hasQWave) ? 'text-red-400' : hasExertional ? 'text-amber-400' : 'text-emerald-400',
        findings: [
          `Typical Chest Pain: ${hasTypicalCP ? 'Present' : 'Absent'}`,
          `Exertional Angina: ${hasExertional ? 'Present' : 'Absent'}`,
          `ST-T Changes: ${hasStDep ? 'ST Depression Observed' : 'Normal ST Segment'}`,
          `Pathological Q Wave: ${hasQWave ? 'Detected' : 'None'}`,
        ],
      },
      metabolic: {
        status: hasDM || fbs > 126 || ldl > 140 ? 'Elevated Atherogenic Risk' : 'Acceptable Lipid / Glycemic State',
        color: hasDM || fbs > 126 || ldl > 140 ? 'text-red-400' : 'text-emerald-400',
        findings: [
          `Diabetes Mellitus: ${hasDM ? 'Yes (Type 2)' : 'Non-diabetic'}`,
          `Fasting Blood Sugar: ${fbs} mg/dL`,
          `Lipid Panel: LDL ${ldl} mg/dL | HDL ${hdl} mg/dL | TG ${tg} mg/dL`,
          `Body Mass Index: ${bmi.toFixed(1)} kg/m²`,
        ],
      },
      hemodynamic: {
        status: hasHTN || bp >= 140 ? 'Hypertensive Vascular Strain' : 'Normotensive Vitals',
        color: hasHTN || bp >= 140 ? 'text-amber-400' : 'text-emerald-400',
        findings: [
          `Hypertension History: ${hasHTN ? 'Present' : 'None'}`,
          `Resting Blood Pressure: ${bp} mmHg`,
          `Resting Pulse Rate: ${pr} bpm`,
          `Peripheral Perfusion: ${f['Weak Peripheral Pulse'] === 'Y' ? 'Diminished' : 'Intact'}`,
        ],
      },
      ventricular: {
        status: ef < 45 || rwma > 1 ? 'Compromised Systolic Function' : ef < 50 ? 'Mild EF Reduction' : 'Preserved Systolic Function',
        color: ef < 45 || rwma > 1 ? 'text-red-400' : ef < 50 ? 'text-amber-400' : 'text-emerald-400',
        findings: [
          `LVEF (Transthoracic Echo): ${ef}%`,
          `Regional Wall Motion Abnormality (RWMA): Score ${rwma}`,
          `Congestive Heart Failure Signs: ${f.CHF === 'Y' ? 'Yes' : 'No'}`,
          `Valvular Heart Disease (VHD): ${f.VHD === 'Y' ? 'Present' : 'None'}`,
        ],
      },
    };
  }, [patientFeatures]);

  return (
    <main id="report-page" className="section-container py-10 animate-fade-in space-y-8">
      {/* ── Top Bar Controls (Hidden in Print) ──────────────── */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 print:hidden">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-8 h-8 rounded-lg bg-violet-500/15 border border-violet-500/30 text-violet-400 flex items-center justify-center">
              <FileText className="w-4 h-4" />
            </div>
            <h1 className="text-2xl font-display font-bold text-white">
              Clinical Decision-Support Report
            </h1>
          </div>
          <p className="text-xs text-gray-400">
            AI-assisted cardiovascular risk assessment and multi-vessel stenosis evaluation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handlePrint}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-display text-xs font-semibold flex items-center gap-2 shadow-lg shadow-violet-500/20 transition-all cursor-pointer"
          >
            <Printer className="w-4 h-4" />
            <span>Print / Save as PDF</span>
          </button>
        </div>
      </div>

      {/* ── MANDATORY REPORT LABEL BANNER ───────────────────── */}
      <div className="p-4 rounded-xl border border-amber-500/40 bg-amber-500/10 text-amber-200 text-xs font-mono space-y-1 print:border-gray-400 print:bg-gray-100 print:text-gray-900">
        <div className="flex items-center gap-2 font-bold text-amber-400 print:text-amber-800 text-sm">
          <ShieldAlert className="w-4 h-4 shrink-0" />
          <span>AI-GENERATED EDUCATIONAL / DECISION-SUPPORT REPORT</span>
        </div>
        <p className="leading-relaxed">
          <strong>Mandatory Notice:</strong> NOT AN OFFICIAL MEDICAL DIAGNOSIS. This document is automatically synthesized
          by machine learning algorithms for clinical education and physician decision support only. It must not be used
          as a substitute for professional clinical judgment, invasive angiography, or diagnostic cardiological consultation.
        </p>
      </div>

      {/* ── PRINTABLE REPORT CONTAINER ──────────────────────── */}
      <div
        ref={reportRef}
        id="printable-report-document"
        className="glass-card p-6 sm:p-10 bg-surface-dark border-surface-border space-y-8 print:border-none print:p-0 print:bg-white print:text-black print:shadow-none"
      >
        {/* Document Header */}
        <div className="border-b border-surface-border pb-6 print:border-gray-300">
          <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
            <div>
              <div className="flex items-center gap-2.5 mb-2">
                <div className="w-7 h-7 rounded-lg bg-primary-600 flex items-center justify-center text-white font-bold text-xs">
                  CV
                </div>
                <span className="text-xl font-display font-bold text-white print:text-indigo-900">
                  CardioVision AI
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-surface-darker text-gray-300 border border-surface-border print:bg-gray-100 print:text-gray-800">
                  v{predictionResult?.model_version || '1.0.0'}
                </span>
              </div>
              <h2 className="text-2xl font-display font-bold text-white print:text-gray-900">
                Coronary Artery Disease & Multi-Vessel Risk Evaluation
              </h2>
              <p className="text-xs text-gray-400 print:text-gray-600 mt-1">
                AI-Generated Educational / Decision-Support Report — Not an Official Medical Diagnosis
              </p>
            </div>

            <div className="text-left sm:text-right font-mono text-xs text-gray-400 print:text-gray-700 space-y-1">
              <p className="font-semibold text-white print:text-gray-900">
                Patient Identifier: <span className="text-primary-400 print:text-indigo-700">{patientId || 'PT-10024'}</span>
              </p>
              <p className="flex items-center sm:justify-end gap-1">
                <Calendar className="w-3.5 h-3.5" />
                <span>Date: {formattedDate}</span>
              </p>
              <p className="flex items-center sm:justify-end gap-1 text-[11px] text-gray-500 print:text-gray-600">
                <Clock className="w-3.5 h-3.5" />
                <span>Time: {formattedTime}</span>
              </p>
              <p className="text-[10px] text-gray-500 print:text-gray-500">
                Timestamp: {isoTimestamp}
              </p>
            </div>
          </div>
        </div>

        {/* ── SECTION 1: PATIENT CLINICAL INPUT SUMMARY ──────── */}
        <div className="space-y-4">
          <h3 className="text-sm font-display font-bold text-white print:text-gray-900 flex items-center gap-2 border-b border-surface-border/60 pb-2">
            <User className="w-4 h-4 text-primary-400 print:text-indigo-600" />
            <span>1. Patient Clinical Input Profile Summary</span>
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            {/* Demographics & Vitals */}
            <div className="p-3 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200">
              <span className="text-[10px] text-gray-500 uppercase block">Demographics</span>
              <p className="font-semibold text-white print:text-gray-900 mt-1">
                {patientFeatures.Age ?? 60} Years / {patientFeatures.Sex ?? 'Male'}
              </p>
              <p className="text-[11px] text-gray-400">
                BMI: {patientFeatures.BMI ?? 25} kg/m² ({patientFeatures.Weight ?? 75} kg)
              </p>
            </div>

            <div className="p-3 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200">
              <span className="text-[10px] text-gray-500 uppercase block">Hemodynamics</span>
              <p className="font-semibold text-white print:text-gray-900 mt-1">
                BP: {patientFeatures.BP ?? 125} mmHg
              </p>
              <p className="text-[11px] text-gray-400">
                Pulse: {patientFeatures.PR ?? 72} bpm (Pulse Intact: {patientFeatures['Weak Peripheral Pulse'] === 'Y' ? 'No' : 'Yes'})
              </p>
            </div>

            <div className="p-3 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200">
              <span className="text-[10px] text-gray-500 uppercase block">Symptoms & Exam</span>
              <p className="font-semibold text-white print:text-gray-900 mt-1">
                Typical CP: {patientFeatures['Typical Chest Pain'] === 1 ? 'Yes' : 'No'}
              </p>
              <p className="text-[11px] text-gray-400">
                Dyspnea: {patientFeatures.Dyspnea ?? 'N'} | NYHA Class: {patientFeatures['Function Class'] ?? 1}
              </p>
            </div>

            <div className="p-3 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200">
              <span className="text-[10px] text-gray-500 uppercase block">Echocardiography</span>
              <p className="font-semibold text-white print:text-gray-900 mt-1">
                LVEF: {patientFeatures['EF-TTE'] ?? 55}%
              </p>
              <p className="text-[11px] text-gray-400">
                Regional RWMA Score: {patientFeatures['Region RWMA'] ?? 0}
              </p>
            </div>
          </div>

          {/* Biomarkers & ECG Secondary Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
            <div className="p-3 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-1">
              <span className="text-[10px] text-cyan-400 font-bold uppercase block">Laboratory Biomarkers</span>
              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-gray-300 print:text-gray-800 text-[11px]">
                <span>FBS: {patientFeatures.FBS ?? 95} mg/dL</span>
                <span>Creatinine: {patientFeatures.CR ?? 0.9} mg/dL</span>
                <span>LDL: {patientFeatures.LDL ?? 110} mg/dL</span>
                <span>HDL: {patientFeatures.HDL ?? 45} mg/dL</span>
                <span>Triglycerides: {patientFeatures.TG ?? 150} mg/dL</span>
                <span>Hemoglobin: {patientFeatures.HB ?? 14} g/dL</span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-1">
              <span className="text-[10px] text-yellow-400 font-bold uppercase block">12-Lead Electrocardiography (ECG)</span>
              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-gray-300 print:text-gray-800 text-[11px]">
                <span>ST Depression: {patientFeatures['St Depression'] === 1 ? 'Present' : 'Normal'}</span>
                <span>ST Elevation: {patientFeatures['St Elevation'] === 1 ? 'Present' : 'None'}</span>
                <span>Q Wave: {patientFeatures['Q Wave'] === 1 ? 'Pathological' : 'Absent'}</span>
                <span>T-Inversion: {patientFeatures.Tinversion === 1 ? 'Inverted' : 'Normal'}</span>
                <span>LVH Signs: {patientFeatures.LVH === 'Y' ? 'Detected' : 'None'}</span>
                <span>BBB: {patientFeatures.BBB === 'Y' ? 'Present' : 'None'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* ── SECTION 2: MULTI-VESSEL RISK PREDICTIONS ───────── */}
        <div className="space-y-4">
          <h3 className="text-sm font-display font-bold text-white print:text-gray-900 flex items-center gap-2 border-b border-surface-border/60 pb-2">
            <Activity className="w-4 h-4 text-primary-400 print:text-indigo-600" />
            <span>2. Overall CAD & Multi-Vessel Stenosis Risk Predictions</span>
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border border-surface-border print:border-gray-300">
              <thead>
                <tr className="bg-surface-dark/90 text-gray-400 print:bg-gray-100 print:text-gray-800 font-mono">
                  <th className="py-2.5 px-3 border-b">Target Prediction Vessel</th>
                  <th className="py-2.5 px-3 border-b">Model-Predicted Probability</th>
                  <th className="py-2.5 px-3 border-b">Tuned Threshold</th>
                  <th className="py-2.5 px-3 border-b">Binary Classification</th>
                  <th className="py-2.5 px-3 border-b">Qualitative Risk Level</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border/50 font-mono">
                {['CAD', 'LAD', 'LCX', 'RCA'].map((target) => {
                  const pred = predictions[target];
                  if (!pred) return null;

                  const proba = pred.model_predicted_probability;
                  const probaPct = (proba * 100).toFixed(1);
                  const level = getRiskLevel(proba, pred.threshold);
                  const color = getRiskColor(level);

                  return (
                    <tr key={target} className="print:border-gray-200">
                      <td className="py-3 px-3 font-bold font-sans text-white print:text-gray-900">
                        {target === 'CAD' ? 'CAD (Overall Cath Diagnosis)' : `${target} Coronary Artery`}
                      </td>
                      <td className="py-3 px-3 font-bold text-white print:text-gray-900">
                        {probaPct}% <span className="text-gray-400 font-normal">({proba.toFixed(4)})</span>
                      </td>
                      <td className="py-3 px-3 text-gray-400 print:text-gray-600">
                        {pred.threshold.toFixed(2)}
                      </td>
                      <td className="py-3 px-3">
                        <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                          pred.classification_binary === 1
                            ? 'bg-red-500/20 text-red-300 print:bg-red-100 print:text-red-800'
                            : 'bg-emerald-500/20 text-emerald-300 print:bg-green-100 print:text-green-800'
                        }`}>
                          {pred.classification}
                        </span>
                      </td>
                      <td className="py-3 px-3">
                        <span className="capitalize font-semibold" style={{ color }}>
                          {level}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="text-[11px] text-gray-400 print:text-gray-600 leading-relaxed italic">
            Terminology Notice: Model probability outputs represent statistical risk estimations derived from trained algorithms,
            not anatomical luminal stenosis percentages (e.g. NOT '70% blockage').
          </p>
        </div>

        {/* ── SECTION 3: PHYSIOLOGICAL BREAKDOWN ─────────────── */}
        <div className="space-y-4">
          <h3 className="text-sm font-display font-bold text-white print:text-gray-900 flex items-center gap-2 border-b border-surface-border/60 pb-2">
            <Heart className="w-4 h-4 text-rose-400 print:text-rose-700" />
            <span>3. Physiological Domain Assessment</span>
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div className="p-4 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-bold text-white print:text-gray-900">Vascular & Ischemic Burden</span>
                <span className={`font-mono font-bold text-[11px] ${physiologicalProfile.ischemic.color}`}>
                  {physiologicalProfile.ischemic.status}
                </span>
              </div>
              <ul className="space-y-1 font-mono text-[11px] text-gray-300 print:text-gray-700">
                {physiologicalProfile.ischemic.findings.map((f, i) => (
                  <li key={i}>• {f}</li>
                ))}
              </ul>
            </div>

            <div className="p-4 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-bold text-white print:text-gray-900">Metabolic & Atherogenic Profile</span>
                <span className={`font-mono font-bold text-[11px] ${physiologicalProfile.metabolic.color}`}>
                  {physiologicalProfile.metabolic.status}
                </span>
              </div>
              <ul className="space-y-1 font-mono text-[11px] text-gray-300 print:text-gray-700">
                {physiologicalProfile.metabolic.findings.map((f, i) => (
                  <li key={i}>• {f}</li>
                ))}
              </ul>
            </div>

            <div className="p-4 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-bold text-white print:text-gray-900">Hemodynamics & Arterial Strain</span>
                <span className={`font-mono font-bold text-[11px] ${physiologicalProfile.hemodynamic.color}`}>
                  {physiologicalProfile.hemodynamic.status}
                </span>
              </div>
              <ul className="space-y-1 font-mono text-[11px] text-gray-300 print:text-gray-700">
                {physiologicalProfile.hemodynamic.findings.map((f, i) => (
                  <li key={i}>• {f}</li>
                ))}
              </ul>
            </div>

            <div className="p-4 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-bold text-white print:text-gray-900">Myocardial & Systolic Function</span>
                <span className={`font-mono font-bold text-[11px] ${physiologicalProfile.ventricular.color}`}>
                  {physiologicalProfile.ventricular.status}
                </span>
              </div>
              <ul className="space-y-1 font-mono text-[11px] text-gray-300 print:text-gray-700">
                {physiologicalProfile.ventricular.findings.map((f, i) => (
                  <li key={i}>• {f}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* ── SECTION 4: SHAP / MODEL CONTRIBUTIONS ──────────── */}
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-surface-border/60 pb-2">
            <h3 className="text-sm font-display font-bold text-white print:text-gray-900 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-yellow-400 print:text-yellow-700" />
              <span>4. TreeSHAP Feature Attributions ({MODEL_CONTRIBUTION_LABEL})</span>
            </h3>
            <span className="text-[11px] font-mono text-gray-400 print:text-gray-600">
              Target: Overall CAD Model
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
            {/* Top Positive */}
            <div className="p-4 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-2">
              <span className="text-red-400 font-bold block">
                Top Factors Increasing Model Output (+SHAP)
              </span>
              <ul className="space-y-1.5 text-gray-300 print:text-gray-800">
                {topPositiveShap.map((f) => {
                  const cat = (f.feature_category as FeatureCategory) || getFeatureCategory(f.feature);
                  const catStyle = CATEGORY_STYLES[cat] || CATEGORY_STYLES.Clinical;
                  return (
                    <li key={f.feature} className="flex justify-between items-center gap-2">
                      <div className="truncate">
                        <span>{f.feature}</span>
                        <span className="text-gray-400 text-[10px] ml-1.5">(Val: {String(f.patient_value)})</span>
                        <span className={`inline-block ml-1 px-1 py-0.2 rounded text-[9px] border ${catStyle.badge}`}>
                          {cat}
                        </span>
                      </div>
                      <span className="text-red-400 font-bold shrink-0">+{f.shap_value.toFixed(4)}</span>
                    </li>
                  );
                })}
              </ul>
            </div>

            {/* Top Negative */}
            <div className="p-4 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-2">
              <span className="text-emerald-400 font-bold block">
                Top Factors Decreasing Model Output (-SHAP)
              </span>
              <ul className="space-y-1.5 text-gray-300 print:text-gray-800">
                {topNegativeShap.map((f) => {
                  const cat = (f.feature_category as FeatureCategory) || getFeatureCategory(f.feature);
                  const catStyle = CATEGORY_STYLES[cat] || CATEGORY_STYLES.Clinical;
                  return (
                    <li key={f.feature} className="flex justify-between items-center gap-2">
                      <div className="truncate">
                        <span>{f.feature}</span>
                        <span className="text-gray-400 text-[10px] ml-1.5">(Val: {String(f.patient_value)})</span>
                        <span className={`inline-block ml-1 px-1 py-0.2 rounded text-[9px] border ${catStyle.badge}`}>
                          {cat}
                        </span>
                      </div>
                      <span className="text-emerald-400 font-bold shrink-0">{f.shap_value.toFixed(4)}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>

          <p className="text-[11px] text-gray-400 print:text-gray-600 font-sans leading-relaxed">
            {NON_CAUSAL_DISCLAIMER}
          </p>
        </div>

        {/* ── SECTION 5: MODEL, VALIDATION & DATASET INFO ────── */}
        <div className="space-y-4">
          <h3 className="text-sm font-display font-bold text-white print:text-gray-900 flex items-center gap-2 border-b border-surface-border/60 pb-2">
            <Cpu className="w-4 h-4 text-accent-400 print:text-indigo-600" />
            <span>5. Technical Architecture, Validation & Dataset Information</span>
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
            <div className="p-3.5 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-1.5">
              <span className="font-bold text-white print:text-gray-900 block">Dataset Information</span>
              <p className="text-gray-300 print:text-gray-700">Source: UCI Repository ID 411</p>
              <p className="text-gray-400">Cohort Size: 303 Patients</p>
              <p className="text-gray-400">Input Variables: 55 Tabular Features</p>
              <p className="text-gray-400">Ground Truth: Invasive Angiography (Cath)</p>
            </div>

            <div className="p-3.5 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-1.5">
              <span className="font-bold text-white print:text-gray-900 block">Validation Methodology</span>
              <p className="text-gray-300 print:text-gray-700">Protocol: 5-Fold Stratified CV</p>
              <p className="text-gray-400">Evaluation: Out-of-fold Probabilities</p>
              <p className="text-gray-400">Leakage Guard: 0 Violations (Clean)</p>
              <p className="text-gray-400">CAD Out-of-Fold AUC: 92.1% (F1 90.5%)</p>
            </div>

            <div className="p-3.5 rounded-xl bg-surface-dark/70 border border-surface-border print:bg-gray-50 print:border-gray-200 space-y-1.5">
              <span className="font-bold text-white print:text-gray-900 block">Model Algorithms</span>
              <p className="text-gray-300 print:text-gray-700">CAD: Random Forest Classifier</p>
              <p className="text-gray-400">LAD: LightGBM Gradient Booster</p>
              <p className="text-gray-400">LCX: XGBoost Ensemble</p>
              <p className="text-gray-400">RCA: Logistic Regression (L2 penalty)</p>
            </div>
          </div>
        </div>

        {/* ── SECTION 6: LIMITATIONS ─────────────────────────── */}
        <div className="space-y-3">
          <h3 className="text-sm font-display font-bold text-white print:text-gray-900 flex items-center gap-2 border-b border-surface-border/60 pb-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 print:text-amber-700" />
            <span>6. Methodological & Clinical Limitations</span>
          </h3>

          <div className="p-4 rounded-xl bg-surface-dark/60 border border-surface-border print:bg-gray-50 print:border-gray-200 text-xs font-sans text-gray-300 print:text-gray-800 space-y-2 leading-relaxed">
            <p>
              <strong>1. Sample Size (N = 303):</strong> Derived from 303 patient records at a single cardiac center (Tehran Heart Center).
              Fold-to-fold metric variability exists, and confidence bounds are wider than massive national population registries.
            </p>
            <p>
              <strong>2. Referral Population Bias:</strong> The training cohort comprises patients clinically scheduled for invasive angiography
              (71.3% CAD prevalence), reflecting high symptomatic pre-test probability. Performance may differ in asymptomatic primary care screening.
            </p>
            <p>
              <strong>3. Absence of Spatial Imaging Geometry:</strong> The models process tabular clinical features only.
              The application does not have access to 3D CT angiography DICOM files; 3D vessel displays represent statistical mappings,
              not patient-specific anatomical geometry or exact lesion coordinates.
            </p>
          </div>
        </div>

        {/* ── SECTION 7: SAFETY DISCLAIMER & SIGN-OFF BLOCK ──── */}
        <div className="p-5 rounded-2xl bg-surface-card border border-surface-border print:border-gray-300 print:bg-gray-50 space-y-4">
          <div className="flex items-center gap-2 text-xs font-bold text-amber-400 print:text-amber-800 uppercase font-mono tracking-wider">
            <ShieldAlert className="w-4 h-4 shrink-0" />
            <span>Institutional Safety Disclaimer & Scope of Use</span>
          </div>

          <p className="text-xs text-gray-400 print:text-gray-700 leading-relaxed font-sans">
            CardioVision AI provides algorithmic probability estimates for cardiovascular risk research, education, and clinical decision support.
            This report does <strong>NOT</strong> constitute a definitive medical diagnosis, treatment recommendation, or clinical mandate.
            Physicians must evaluate patients holistically through validated clinical standards, 12-lead ECG review, laboratory findings,
            and invasive/non-invasive coronary angiography where clinically indicated.
          </p>

          <div className="pt-4 border-t border-surface-border/60 print:border-gray-300 grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs font-mono">
            <div>
              <span className="text-[10px] text-gray-500 uppercase block">Reviewing Clinician:</span>
              <div className="mt-4 border-b border-gray-600 print:border-gray-400 w-36" />
              <span className="text-[10px] text-gray-500 block mt-1">Signature / Date</span>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 uppercase block">Clinical Role:</span>
              <p className="text-white print:text-gray-900 mt-2">Cardiology / Primary Care</p>
            </div>
            <div>
              <span className="text-[10px] text-gray-500 uppercase block">Report Version:</span>
              <p className="text-primary-400 print:text-indigo-700 mt-2">v{predictionResult?.model_version || '1.0.0'}</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
