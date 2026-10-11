/**
 * HomePage – The landing page.
 *
 * "Zero-effort entry point" as specified in instructions.md Phase 8.
 * Single compelling headline + upload form.
 */

import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { useMutation } from '@tanstack/react-query';
import {
  Sparkles, Upload, Link, ArrowRight, FileText,
  CheckCircle2, Zap, Target, Shield, X
} from 'lucide-react';

import useAuthStore from '../store/authStore';
import { jobsApi } from '../api/client';
import Spinner from '../components/Spinner';

// ── Feature highlights ───────────────────────────────────────────────────────
const FEATURES = [
  { icon: Target,       title: 'ATS Score',       desc: 'Know exactly how well you match with a data-driven score.' },
  { icon: Zap,          title: 'AI Suggestions',  desc: 'Get specific, actionable tips to improve your alignment.' },
  { icon: FileText,     title: 'Cover Letter',     desc: 'Generate a personalized, professional cover letter in seconds.' },
  { icon: Shield,       title: 'Private & Secure', desc: 'Your data is encrypted and never used for model training.' },
];

// ── How it works ────────────────────────────────────────────────────────────
const STEPS = [
  { num: '01', title: 'Upload Your Resume',   desc: 'Drop your PDF. We handle the rest.' },
  { num: '02', title: 'Paste the Job Link',   desc: 'Any public job description URL.' },
  { num: '03', title: 'Get Instant Insights', desc: 'Score, gaps, suggestions — all in one place.' },
];

export default function HomePage() {
  const { isAuthenticated } = useAuthStore();
  const navigate = useNavigate();

  const [file, setFile]   = useState(null);
  const [url, setUrl]     = useState('');
  const [error, setError] = useState('');

  // ── File drop ──────────────────────────────────────────────────────────────
  const onDrop = useCallback((acceptedFiles, rejections) => {
    setError('');
    if (rejections.length > 0) {
      setError('Please upload a PDF file (max 10MB).');
      return;
    }
    setFile(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
  });

  // ── Submit mutation ────────────────────────────────────────────────────────
  const { mutate: submitAnalysis, isPending } = useMutation({
    mutationFn: () => jobsApi.submitAnalysis(file, url),
    onSuccess: ({ data }) => {
      navigate(`/dashboard/${data.job_id}`);
    },
    onError: (err) => {
      const msg = err.response?.data?.detail || 'Something went wrong. Please try again.';
      setError(msg);
    },
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (!isAuthenticated()) {
      navigate('/login', { state: { from: '/' } });
      return;
    }
    if (!file)  { setError('Please upload your resume PDF.'); return; }
    if (!url.trim()) { setError('Please enter a job description URL.'); return; }

    submitAnalysis();
  };

  const canSubmit = file && url.trim() && !isPending;

  return (
    <div className="hero-gradient-bg min-h-screen">

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section className="page-container pt-20 pb-16">
        <div className="max-w-3xl mx-auto text-center">

          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8 animate-fade-in"
            style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.25)' }}
          >
            <Sparkles className="w-4 h-4" style={{ color: 'var(--color-accent-indigo)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--color-accent-indigo)' }}>
              Powered by Mistral AI & Llama 2
            </span>
          </div>

          {/* Headline */}
          <h1
            className="text-5xl sm:text-6xl font-black mb-6 leading-tight text-balance animate-fade-in-up"
            style={{ fontFamily: 'var(--font-display)', animationDelay: '100ms', animationFillMode: 'both' }}
          >
            Stop guessing.
            <br />
            <span className="gradient-text">Know your match score.</span>
          </h1>

          <p
            className="text-lg sm:text-xl mb-12 max-w-2xl mx-auto text-balance animate-fade-in-up"
            style={{ color: 'var(--color-text-secondary)', animationDelay: '200ms', animationFillMode: 'both' }}
          >
            AI-powered analysis compares your resume to any job description, gives you an ATS alignment score, actionable suggestions, and generates a personalized cover letter — in under a minute.
          </p>

          {/* ── Upload Form ─────────────────────────────────────────────────── */}
          <div
            className="card-elevated text-left animate-fade-in-up"
            style={{ animationDelay: '300ms', animationFillMode: 'both' }}
          >
            <form onSubmit={handleSubmit} id="analysis-form">

              {/* Resume dropzone */}
              <div className="mb-5">
                <label className="input-label">Your Resume (PDF)</label>
                <div
                  {...getRootProps()}
                  className="relative rounded-xl p-8 text-center cursor-pointer transition-all duration-200"
                  style={{
                    background: isDragActive
                      ? 'rgba(99,102,241,0.12)'
                      : file
                      ? 'rgba(16,185,129,0.08)'
                      : 'rgba(99,102,241,0.04)',
                    border: isDragActive
                      ? '2px dashed var(--color-accent-indigo)'
                      : file
                      ? '2px dashed rgba(16,185,129,0.5)'
                      : '2px dashed var(--color-border-default)',
                  }}
                  id="resume-dropzone"
                >
                  <input {...getInputProps()} id="resume-file-input" />

                  {file ? (
                    <div className="flex items-center justify-center gap-3">
                      <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{ background: 'rgba(16,185,129,0.15)' }}
                      >
                        <FileText className="w-5 h-5" style={{ color: '#10b981' }} />
                      </div>
                      <div className="text-left">
                        <p className="font-semibold text-sm" style={{ color: '#10b981' }}>
                          {file.name}
                        </p>
                        <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                          {(file.size / 1024).toFixed(1)} KB · PDF
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); setFile(null); }}
                        className="ml-auto p-1.5 rounded-lg hover:bg-red-500/10 transition-colors"
                        aria-label="Remove file"
                      >
                        <X className="w-4 h-4" style={{ color: '#f43f5e' }} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="w-12 h-12 rounded-2xl gradient-brand mx-auto mb-4 flex items-center justify-center animate-float">
                        <Upload className="w-6 h-6 text-white" />
                      </div>
                      <p className="font-semibold mb-1" style={{ color: 'var(--color-text-primary)' }}>
                        {isDragActive ? 'Drop your PDF here' : 'Drag & drop your resume'}
                      </p>
                      <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                        or <span style={{ color: 'var(--color-accent-indigo)' }}>click to browse</span> · PDF only · max 10MB
                      </p>
                    </>
                  )}
                </div>
              </div>

              {/* Job URL input */}
              <div className="mb-6">
                <label className="input-label" htmlFor="job-url-input">Job Description URL</label>
                <div className="relative">
                  <Link className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none"
                    style={{ color: 'var(--color-text-muted)' }}
                  />
                  <input
                    id="job-url-input"
                    type="url"
                    className="input"
                    style={{ paddingLeft: '2.5rem' }}
                    placeholder="https://careers.company.com/job/data-scientist-123"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    required
                  />
                </div>
              </div>

              {/* Error */}
              {error && (
                <div
                  className="flex items-start gap-2 mb-5 p-3 rounded-xl animate-fade-in"
                  style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.2)' }}
                >
                  <span className="text-sm" style={{ color: '#f43f5e' }}>⚠️ {error}</span>
                </div>
              )}

              {/* Submit */}
              <button
                type="submit"
                className="btn btn-primary btn-lg w-full"
                disabled={!canSubmit}
                id="analyze-btn"
              >
                {isPending ? (
                  <><Spinner size="sm" className="border-white/30 border-t-white" /> Analyzing…</>
                ) : (
                  <><Sparkles className="w-5 h-5" /> Analyze My Resume <ArrowRight className="w-5 h-5" /></>
                )}
              </button>

              {!isAuthenticated() && (
                <p className="text-center text-xs mt-3" style={{ color: 'var(--color-text-muted)' }}>
                  You'll be asked to sign in before we run the analysis.
                </p>
              )}
            </form>
          </div>
        </div>
      </section>

      {/* ── How It Works ───────────────────────────────────────────────── */}
      <section className="page-container pb-16">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-2xl font-bold" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-text-primary)' }}>
              How It Works
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {STEPS.map((step, i) => (
              <div
                key={step.num}
                className="card text-center animate-fade-in-up"
                style={{ animationDelay: `${i * 100}ms`, animationFillMode: 'both' }}
              >
                <div
                  className="text-3xl font-black mb-3 gradient-text"
                  style={{ fontFamily: 'var(--font-display)' }}
                >
                  {step.num}
                </div>
                <h3 className="font-bold mb-2 text-base" style={{ color: 'var(--color-text-primary)' }}>
                  {step.title}
                </h3>
                <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────────────────── */}
      <section className="page-container pb-24">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 max-w-5xl mx-auto">
          {FEATURES.map((feat, i) => (
            <div
              key={feat.title}
              className="card text-center animate-fade-in-up"
              style={{ animationDelay: `${i * 80}ms`, animationFillMode: 'both' }}
            >
              <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center mx-auto mb-3">
                <feat.icon className="w-5 h-5 text-white" />
              </div>
              <h3 className="font-bold text-sm mb-1.5" style={{ color: 'var(--color-text-primary)' }}>
                {feat.title}
              </h3>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
                {feat.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
