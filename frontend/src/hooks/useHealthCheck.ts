/**
 * CardioVision AI — useHealthCheck Hook
 * ========================================
 * Custom hook for polling backend health status.
 */

import { useState, useEffect, useCallback } from 'react';
import { checkHealth } from '@/services/api';
import type { HealthResponse } from '@/types';

interface UseHealthCheckReturn {
  health: HealthResponse | null;
  status: 'idle' | 'checking' | 'ok' | 'error';
  error: string | null;
  refresh: () => void;
}

export function useHealthCheck(intervalMs = 30_000): UseHealthCheckReturn {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<'idle' | 'checking' | 'ok' | 'error'>('checking');
  const [error, setError]   = useState<string | null>(null);

  const check = useCallback(async () => {
    try {
      const data = await checkHealth();
      setHealth(data);
      setStatus(data.status === 'ok' ? 'ok' : 'error');
      setError(null);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Backend unreachable');
    }
  }, []);

  useEffect(() => {
    void check();
    const id = setInterval(() => { void check(); }, intervalMs);
    return () => clearInterval(id);
  }, [check, intervalMs]);

  return { health, status, error, refresh: check };
}
