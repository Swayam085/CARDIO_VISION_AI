/**
 * CardioVision AI — Analyze Page (/analyze)
 * ==========================================
 * Clinical feature input form page supporting 55 variables and 1-click presets.
 */

import { FlaskConical } from 'lucide-react';
import { SafetyDisclaimer } from '@/components/ui/SafetyDisclaimer';
import { ClinicalForm } from '@/features/analysis/ClinicalForm';

export function AnalyzePage() {
  return (
    <main id="analyze-page" className="section-container py-10 animate-fade-in">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-accent-500/15 border border-accent-500/30 mb-3">
            <FlaskConical className="w-6 h-6 text-accent-400" aria-hidden="true" />
          </div>
          <h1 className="text-3xl font-display font-bold gradient-text-accent mb-2">
            Clinical Feature Analysis
          </h1>
          <p className="text-gray-400 text-sm max-w-xl mx-auto leading-relaxed">
            Provide patient demographics, vitals, symptoms, ECG, lab parameters, and echo metrics.
            <span className="text-amber-400 font-medium"> Vessel labels (LAD, LCX, RCA, Cath) are strictly excluded from input features.</span>
          </p>
        </div>

        <SafetyDisclaimer className="mb-6" />

        {/* Form Container */}
        <ClinicalForm />
      </div>
    </main>
  );
}
