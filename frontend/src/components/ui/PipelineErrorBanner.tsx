/**
 * CardioVision AI — Pipeline Error Banner Component
 * ==================================================
 * Complies with Requirement 9:
 * Displays structured, actionable error states:
 *   - API error (connection refused / server 500)
 *   - Validation error (422 / out-of-bounds or wrong types)
 *   - Data leakage violation (LAD/LCX/RCA in inputs)
 *   - Missing model artifact (503 / models not trained)
 *   - Malformed response (missing prediction task keys)
 */

import {
  AlertCircle,
  WifiOff,
  ShieldAlert,
  FileQuestion,
  RefreshCw,
  Box,
} from 'lucide-react';

export type PipelineErrorType =
  | 'api_error'
  | 'validation_error'
  | 'leakage_error'
  | 'missing_artifact'
  | '3d_error'
  | 'malformed_response';

export interface PipelineErrorInfo {
  type: PipelineErrorType;
  title: string;
  message: string;
  fieldErrors?: Array<{ field?: string; message: string }>;
}

interface Props {
  error: string | PipelineErrorInfo | null;
  onRetry?: () => void;
  className?: string;
}

function parseError(rawError: any): PipelineErrorInfo {
  if (typeof rawError === 'object' && rawError?.type) {
    return rawError as PipelineErrorInfo;
  }

  const errStr = typeof rawError === 'string' ? rawError : String(rawError?.message || '');

  // Detect Data Leakage
  if (errStr.toLowerCase().includes('leakage') || errStr.toLowerCase().includes('forbidden')) {
    return {
      type: 'leakage_error',
      title: 'Data Leakage Guard Activated (422)',
      message: 'Prohibited target columns (LAD, LCX, RCA, Cath) detected in input features. The clinical analysis pipeline strictly excludes downstream target labels to prevent clinical bias.',
    };
  }

  // Detect Missing Artifact / 503
  if (errStr.includes('503') || errStr.toLowerCase().includes('artifact') || errStr.toLowerCase().includes('not trained')) {
    return {
      type: 'missing_artifact',
      title: 'ML Model Artifacts Unavailable (503)',
      message: 'The trained model joblib files are missing or uninitialized on the server. Please run the training pipeline (python ml/src/train.py) or wait for server startup.',
    };
  }

  // Detect Validation / 422
  if (errStr.includes('422') || errStr.toLowerCase().includes('validation') || errStr.toLowerCase().includes('unprocessable')) {
    return {
      type: 'validation_error',
      title: 'Clinical Input Validation Error (422)',
      message: errStr || 'One or more patient features are out of valid physiological bounds or have invalid categorical types.',
    };
  }

  // Detect Malformed Response
  if (errStr.toLowerCase().includes('malformed') || errStr.toLowerCase().includes('missing task')) {
    return {
      type: 'malformed_response',
      title: 'Malformed API Response',
      message: 'The backend response was incomplete or missing one or more required prediction tasks (CAD, LAD, LCX, RCA).',
    };
  }

  // Detect 3D WebGL
  if (errStr.toLowerCase().includes('3d') || errStr.toLowerCase().includes('webgl') || errStr.toLowerCase().includes('glb')) {
    return {
      type: '3d_error',
      title: '3D Anatomical Visualization Error',
      message: 'Failed to initialize WebGL context or load the BodyParts3D GLB model. Falling back to schematic vessel inspector.',
    };
  }

  // Default to API connection error
  return {
    type: 'api_error',
    title: 'Backend Communication Error',
    message: errStr || 'Unable to establish connection to the FastAPI backend service. Verify that the server is active on port 8000.',
  };
}

export function PipelineErrorBanner({ error, onRetry, className = '' }: Props) {
  if (!error) return null;

  const parsed = parseError(error);

  const ICONS: Record<PipelineErrorType, any> = {
    api_error: WifiOff,
    validation_error: AlertCircle,
    leakage_error: ShieldAlert,
    missing_artifact: FileQuestion,
    '3d_error': Box,
    malformed_response: AlertCircle,
  };

  const COLORS: Record<PipelineErrorType, { bg: string; border: string; text: string; badge: string }> = {
    api_error: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-300', badge: 'bg-red-500/20 text-red-400' },
    validation_error: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-300', badge: 'bg-amber-500/20 text-amber-400' },
    leakage_error: { bg: 'bg-rose-500/15', border: 'border-rose-500/40', text: 'text-rose-200', badge: 'bg-rose-500/30 text-rose-300' },
    missing_artifact: { bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-300', badge: 'bg-yellow-500/20 text-yellow-400' },
    '3d_error': { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-300', badge: 'bg-purple-500/20 text-purple-400' },
    malformed_response: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-300', badge: 'bg-red-500/20 text-red-400' },
  };

  const style = COLORS[parsed.type];
  const Icon = ICONS[parsed.type];

  return (
    <div
      role="alert"
      className={`p-4 rounded-xl border ${style.bg} ${style.border} ${style.text} flex flex-col sm:flex-row items-start justify-between gap-4 text-xs font-mono ${className}`}
    >
      <div className="flex items-start gap-3">
        <Icon className="w-5 h-5 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-bold text-white text-sm">{parsed.title}</span>
            <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${style.badge}`}>
              {parsed.type.replace('_', ' ')}
            </span>
          </div>
          <p className="leading-relaxed text-gray-300">{parsed.message}</p>
        </div>
      </div>

      {onRetry && (
        <button
          onClick={onRetry}
          className="px-3.5 py-1.5 rounded-lg bg-surface-darker/80 hover:bg-surface-dark border border-surface-border text-white text-xs font-semibold flex items-center gap-1.5 shrink-0 transition-all"
        >
          <RefreshCw className="w-3.5 h-3.5 text-accent-400" />
          <span>Retry</span>
        </button>
      )}
    </div>
  );
}
