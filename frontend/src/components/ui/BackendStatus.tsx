/**
 * CardioVision AI — Backend Status Badge
 * =========================================
 * Polls /health and shows live backend status.
 */

import { useEffect, useState } from 'react';
import { checkHealth } from '@/services/api';
import type { HealthResponse } from '@/types';

type Status = 'checking' | 'ok' | 'error';

export function BackendStatus() {
  const [status, setStatus] = useState<Status>('checking');
  const [info, setInfo]     = useState<HealthResponse | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const data = await checkHealth();
        setInfo(data);
        setStatus(data.status === 'ok' ? 'ok' : 'error');
      } catch {
        setStatus('error');
      }
    };
    void check();
    const interval = setInterval(() => { void check(); }, 30_000);
    return () => clearInterval(interval);
  }, []);

  const dot: Record<Status, string> = {
    checking: 'bg-yellow-500 animate-pulse',
    ok:       'bg-green-500',
    error:    'bg-red-500',
  };
  const label: Record<Status, string> = {
    checking: 'Connecting…',
    ok:       `API v${info?.version ?? '?'}`,
    error:    'Backend offline',
  };

  return (
    <div
      id="backend-status"
      className="flex items-center gap-1.5 text-xs text-gray-500"
      title={status === 'ok' ? `Uptime: ${info?.uptime_seconds?.toFixed(0)}s` : 'Backend not reachable'}
      role="status"
      aria-live="polite"
    >
      <span className={`w-2 h-2 rounded-full shrink-0 ${dot[status]}`} aria-hidden="true" />
      <span>{label[status]}</span>
    </div>
  );
}
