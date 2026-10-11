/**
 * API Client – Axios instance wired to the FastAPI backend.
 * All requests go through this client so auth headers are injected
 * automatically from the Zustand auth store.
 */

import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || '';  // Vite proxy handles /api/* → localhost:8000

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Request interceptor: inject JWT ─────────────────────────────────────────
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('intellicover_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Response interceptor: handle 401 globally ───────────────────────────────
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('intellicover_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;

// ── Auth ─────────────────────────────────────────────────────────────────────
export const authApi = {
  register: (data) => apiClient.post('/api/v1/auth/register', data),
  login: (data)    => apiClient.post('/api/v1/auth/login', data),
  me: ()           => apiClient.get('/api/v1/auth/me'),
};

// ── Jobs ─────────────────────────────────────────────────────────────────────
export const jobsApi = {
  /**
   * Submit resume PDF + job URL.
   * Uses FormData (multipart/form-data).
   */
  submitAnalysis: (file, url) => {
    const form = new FormData();
    form.append('resume', file);
    form.append('url', url);
    return apiClient.post('/api/v1/jobs/analyze', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  /** Poll for job status and results. */
  getJobStatus: (jobId) => apiClient.get(`/api/v1/jobs/${jobId}/status`),

  /** Retry a failed analysis with raw text */
  retryAnalysis: (jobId, text) =>
    apiClient.post(`/api/v1/jobs/${jobId}/retry`, { job_description_text: text }),

  /** Trigger cover letter generation. */
  generateCoverLetter: (jobId) =>
    apiClient.post(`/api/v1/jobs/${jobId}/generate-cover-letter`),

  /** Poll for cover letter status. */
  getCoverLetterStatus: (jobId) =>
    apiClient.get(`/api/v1/jobs/${jobId}/cover-letter/status`),
};
