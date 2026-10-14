/**
 * CardioVision AI — Safety Disclaimer Component
 * ================================================
 * Reusable clinical safety notice that must appear on all prediction-related pages.
 *
 * Displays two required notices:
 * 1. General clinical safety notice (not a diagnostic tool)
 * 2. 3D visualization notice (vessel-level predictions, not exact lesion coordinates)
 */

import { AlertTriangle, Info } from 'lucide-react';
import { clsx } from 'clsx';

interface SafetyDisclaimerProps {
  /** Show the 3D visualization-specific notice */
  showVizNote?: boolean;
  /** Compact single-line variant */
  compact?: boolean;
  /** Extra className */
  className?: string;
}

const CLINICAL_SAFETY_NOTICE =
  'Clinical Safety Notice: This application provides model-based estimates for educational ' +
  'and decision-support purposes only. It is not a medical diagnosis and does not replace ' +
  'professional clinical evaluation, diagnostic imaging, or advice from a qualified healthcare professional.';

const VIZ_SAFETY_NOTICE =
  'The 3D visualization represents vessel-level model predictions and does not represent ' +
  'exact lesion coordinates, CT/MRI segmentation, or pixel-level anatomical data.';

export function SafetyDisclaimer({
  showVizNote = false,
  compact = false,
  className,
}: SafetyDisclaimerProps) {
  if (compact) {
    return (
      <div
        id="safety-disclaimer-compact"
        role="note"
        aria-label="Clinical safety notice"
        className={clsx(
          'safety-disclaimer flex items-start gap-2 px-3 py-2',
          className,
        )}
      >
        <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" aria-hidden="true" />
        <p className="text-xs text-red-300/80 leading-relaxed">{CLINICAL_SAFETY_NOTICE}</p>
      </div>
    );
  }

  return (
    <div
      id="safety-disclaimer"
      role="note"
      aria-label="Clinical safety notice"
      className={clsx('safety-disclaimer p-4 space-y-3', className)}
    >
      {/* Clinical notice */}
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">
          <AlertTriangle className="w-5 h-5 text-red-400" aria-hidden="true" />
        </div>
        <div>
          <p className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-1">
            Clinical Safety Notice
          </p>
          <p className="text-sm text-red-300/80 leading-relaxed">{CLINICAL_SAFETY_NOTICE}</p>
        </div>
      </div>

      {/* 3D viz notice — shown on visualization pages */}
      {showVizNote && (
        <div className="flex items-start gap-3 pt-2 border-t border-red-900/20">
          <div className="flex-shrink-0 mt-0.5">
            <Info className="w-5 h-5 text-amber-400" aria-hidden="true" />
          </div>
          <div>
            <p className="text-xs font-semibold text-amber-400 uppercase tracking-wider mb-1">
              3D Visualization Note
            </p>
            <p className="text-sm text-amber-300/80 leading-relaxed">{VIZ_SAFETY_NOTICE}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export { CLINICAL_SAFETY_NOTICE, VIZ_SAFETY_NOTICE };
