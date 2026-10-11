/**
 * useJobPolling – Custom hook that polls the job status endpoint
 * every 3 seconds until the job reaches a terminal state.
 *
 * Uses TanStack Query's refetchInterval for automatic polling.
 */

import { useQuery } from '@tanstack/react-query';
import { jobsApi } from '../api/client';

const TERMINAL_STATES = ['SUCCESS', 'FAILURE'];
const POLL_INTERVAL_MS = 3000;

export function useJobPolling(jobId, options = {}) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: async () => {
      const { data } = await jobsApi.getJobStatus(jobId);
      return data;
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Stop polling once we hit a terminal state
      if (TERMINAL_STATES.includes(status)) return false;
      return POLL_INTERVAL_MS;
    },
    staleTime: 0,
    ...options,
  });
}

/**
 * useCoverLetterPolling – Polls cover letter status every 3 seconds.
 */
export function useCoverLetterPolling(jobId, options = {}) {
  return useQuery({
    queryKey: ['cover-letter', jobId],
    queryFn: async () => {
      const { data } = await jobsApi.getCoverLetterStatus(jobId);
      return data;
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (TERMINAL_STATES.includes(status)) return false;
      return POLL_INTERVAL_MS;
    },
    staleTime: 0,
    ...options,
  });
}
