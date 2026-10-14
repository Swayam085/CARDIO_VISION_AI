/**
 * CardioVision AI — Physiological Profile Component
 * ==================================================
 * Complies with Requirement 5:
 * Displays supplied patient measurements alongside model inputs:
 * Age, BP, Pulse, Lipids, ECG variables, Echo variables.
 * CRITICAL RULE: Only shows values that were actually supplied by the user/preset.
 */

import { useMemo } from 'react';
import {
  User,
  HeartPulse,
  Activity,
  TestTube,
  Flame,
} from 'lucide-react';

interface Props {
  patientFeatures: Record<string, any>;
  patientId: string;
  className?: string;
}

interface FeatureDisplayGroup {
  name: string;
  icon: any;
  accentColor: string;
  items: { label: string; key: string; format?: (v: any) => string; unit?: string }[];
}

const PROFILE_GROUPS: FeatureDisplayGroup[] = [
  {
    name: 'Demographics & Vitals',
    icon: User,
    accentColor: 'text-blue-400',
    items: [
      { label: 'Age', key: 'Age', unit: 'years' },
      { label: 'Sex', key: 'Sex' },
      { label: 'Blood Pressure (BP)', key: 'BP', unit: 'mmHg' },
      { label: 'Pulse Rate (PR)', key: 'PR', unit: 'bpm' },
      { label: 'BMI', key: 'BMI', format: (v) => Number(v).toFixed(1) },
      { label: 'Weight', key: 'Weight', unit: 'kg' },
      { label: 'Length / Height', key: 'Length', unit: 'cm' },
    ],
  },
  {
    name: 'Cardiac Symptoms & Exam',
    icon: Flame,
    accentColor: 'text-purple-400',
    items: [
      { label: 'Typical Chest Pain', key: 'Typical Chest Pain', format: (v) => (v === 1 ? 'Yes' : 'No') },
      { label: 'Exertional CP', key: 'Exertional CP' },
      { label: 'Dyspnea', key: 'Dyspnea' },
      { label: 'Function Class', key: 'Function Class', format: (v) => `Class ${v}` },
      { label: 'Hypertension (HTN)', key: 'HTN', format: (v) => (v === 1 ? 'Yes' : 'No') },
      { label: 'Diabetes Mellitus (DM)', key: 'DM', format: (v) => (v === 1 ? 'Yes' : 'No') },
      { label: 'Current Smoker', key: 'Current Smoker', format: (v) => (v === 1 ? 'Yes' : 'No') },
      { label: 'Family History (FH)', key: 'FH', format: (v) => (v === 1 ? 'Yes' : 'No') },
    ],
  },
  {
    name: 'ECG Findings',
    icon: Activity,
    accentColor: 'text-yellow-400',
    items: [
      { label: 'ST Depression', key: 'St Depression', format: (v) => (v === 1 ? 'Yes' : 'No') },
      { label: 'ST Elevation', key: 'St Elevation', format: (v) => (v === 1 ? 'Yes' : 'No') },
      { label: 'T-Wave Inversion', key: 'Tinversion', format: (v) => (v === 1 ? 'Yes' : 'No') },
      { label: 'Q-Wave', key: 'Q Wave', format: (v) => (v === 1 ? 'Yes' : 'No') },
      { label: 'LVH (Left Ventricular Hypertrophy)', key: 'LVH' },
      { label: 'Poor R Progression', key: 'Poor R Progression' },
    ],
  },
  {
    name: 'Laboratory Lipid & Metabolic Panel',
    icon: TestTube,
    accentColor: 'text-cyan-400',
    items: [
      { label: 'Fasting Blood Sugar (FBS)', key: 'FBS', unit: 'mg/dL' },
      { label: 'Triglycerides (TG)', key: 'TG', unit: 'mg/dL' },
      { label: 'LDL Cholesterol', key: 'LDL', unit: 'mg/dL' },
      { label: 'HDL Cholesterol', key: 'HDL', unit: 'mg/dL' },
      { label: 'Creatinine (CR)', key: 'CR', unit: 'mg/dL' },
      { label: 'BUN', key: 'BUN', unit: 'mg/dL' },
      { label: 'Hemoglobin (HB)', key: 'HB', unit: 'g/dL' },
      { label: 'ESR', key: 'ESR', unit: 'mm/hr' },
    ],
  },
  {
    name: 'Echocardiography (Echo)',
    icon: HeartPulse,
    accentColor: 'text-emerald-400',
    items: [
      { label: 'Ejection Fraction (EF-TTE)', key: 'EF-TTE', unit: '%' },
      {
        label: 'Regional Wall Motion Abnormality (RWMA)',
        key: 'Region RWMA',
        format: (v) => `${v} segment${Number(v) === 1 ? '' : 's'}`,
      },
    ],
  },
];

export function PhysiologicalProfileCard({
  patientFeatures,
  patientId,
  className = '',
}: Props) {
  // Filter only values that actually exist in patientFeatures
  const suppliedGroups = useMemo(() => {
    return PROFILE_GROUPS.map((group) => {
      const activeItems = group.items.filter((item) => {
        const val = patientFeatures[item.key];
        return val !== undefined && val !== null && val !== '';
      });
      return {
        ...group,
        activeItems,
      };
    }).filter((group) => group.activeItems.length > 0);
  }, [patientFeatures]);

  const totalSupplied = useMemo(() => {
    return Object.keys(patientFeatures).filter(
      (k) => patientFeatures[k] !== undefined && patientFeatures[k] !== null && patientFeatures[k] !== ''
    ).length;
  }, [patientFeatures]);

  return (
    <div
      id="physiological-profile-card"
      className={`glass-card p-6 border-surface-border space-y-6 ${className}`}
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-surface-border/70 pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full bg-primary-400" />
            <h3 className="text-xl font-display font-bold text-white">
              Patient Physiological Profile
            </h3>
          </div>
          <p className="text-xs text-gray-400">
            Actual clinical measurements supplied for patient{' '}
            <code className="text-primary-300 font-mono">{patientId}</code>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded bg-primary-500/10 border border-primary-500/30 text-primary-400 text-xs font-mono font-semibold">
            {totalSupplied} Supplied Features
          </span>
          <span className="px-2.5 py-1 rounded bg-surface-dark border border-surface-border text-gray-400 text-xs font-mono">
            UCI ID 411 Format
          </span>
        </div>
      </div>

      {/* Grid of physiological categories */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {suppliedGroups.map((grp) => {
          const Icon = grp.icon;
          return (
            <div
              key={grp.name}
              className="p-4 rounded-xl bg-surface-dark/70 border border-surface-border/60 flex flex-col justify-between space-y-3"
            >
              <div className="flex items-center gap-2 border-b border-surface-border/40 pb-2">
                <Icon className={`w-4 h-4 ${grp.accentColor}`} />
                <span className="text-xs font-mono font-bold text-gray-200">
                  {grp.name}
                </span>
              </div>

              <div className="space-y-2 text-xs font-mono flex-1">
                {grp.activeItems.map((item) => {
                  const rawVal = patientFeatures[item.key];
                  const displayVal = item.format ? item.format(rawVal) : String(rawVal);
                  return (
                    <div
                      key={item.key}
                      className="flex items-center justify-between py-1 border-b border-surface-border/20 last:border-0"
                    >
                      <span className="text-gray-400 truncate pr-2" title={item.label}>
                        {item.label}
                      </span>
                      <span className="text-white font-semibold shrink-0">
                        {displayVal}
                        {item.unit ? ` ${item.unit}` : ''}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
