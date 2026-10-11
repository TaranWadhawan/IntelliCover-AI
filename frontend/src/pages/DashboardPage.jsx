/**
 * DashboardPage – The core results page.
 *
 * Polling lifecycle:
 *   1. PENDING / IN_PROGRESS → show ProcessingSteps + live status
 *   2. SUCCESS               → show ATSScoreCard + KeywordMatch + Suggestions + CoverLetter CTA
 *   3. FAILURE               → show error card
 *
 * Cover letter lifecycle:
 *   - User clicks "Generate Cover Letter"
 *   - POST → triggers Celery task
 *   - Poll until SUCCESS → show CoverLetterEditor
 */

import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import {
  ArrowLeft, FileText, Sparkles, AlertCircle,
  RefreshCw, Clock, Building2, Briefcase
} from 'lucide-react';

import { useJobPolling, useCoverLetterPolling } from '../hooks/useJobPolling';
import { jobsApi } from '../api/client';

import ATSScoreCard from '../components/ATSScoreCard';
import KeywordMatchPanel from '../components/KeywordMatchPanel';
import SuggestionsPanel from '../components/SuggestionsPanel';
import ProcessingSteps from '../components/ProcessingSteps';
import CoverLetterEditor from '../components/CoverLetterEditor';
import Spinner from '../components/Spinner';
import JDRetryForm from '../components/JDRetryForm';

// ── Skeleton loaders ─────────────────────────────────────────────────────────
function SkeletonCard({ height = 300 }) {
  return (
    <div
      className="skeleton rounded-2xl"
      style={{ height, width: '100%' }}
    />
  );
}

// ── Status badge ──────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
  const config = {
    PENDING:     { label: 'Queued',      cls: 'badge-info' },
    IN_PROGRESS: { label: 'Processing',  cls: 'badge-warning' },
    SUCCESS:     { label: 'Complete',    cls: 'badge-success' },
    FAILURE:     { label: 'Failed',      cls: 'badge-error' },
  };
  const c = config[status] || config.PENDING;
  return <span className={`badge ${c.cls}`}>{c.label}</span>;
}

// ── Cover Letter Generation Section ──────────────────────────────────────────
function CoverLetterSection({ jobId, parsedJd }) {
  const [triggered, setTriggered] = useState(false);

  const { mutate: generateCL, isPending: isGenerating } = useMutation({
    mutationFn: () => jobsApi.generateCoverLetter(jobId),
    onSuccess: () => setTriggered(true),
    onError: (err) => {
      const msg = err.response?.data?.detail;
      // If a cover letter already exists (409), just trigger polling
      if (err.response?.status === 409) {
        setTriggered(true);
      } else {
        console.error('Cover letter generation error:', msg);
      }
    },
  });

  const {
    data: clData,
    isLoading: isClLoading,
  } = useCoverLetterPolling(triggered ? jobId : null);

  // Not triggered yet — show the CTA button
  if (!triggered) {
    return (
      <div className="card-elevated text-center py-10 animate-fade-in-up delay-400">
        <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5 animate-float"
          style={{ background: 'rgba(139,92,246,0.15)', border: '1px solid rgba(139,92,246,0.25)' }}
        >
          <FileText className="w-8 h-8" style={{ color: '#8b5cf6' }} />
        </div>
        <h3 className="text-xl font-bold mb-2" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-text-primary)' }}>
          Generate Your Cover Letter
        </h3>
        <p className="text-sm mb-6 max-w-sm mx-auto" style={{ color: 'var(--color-text-muted)' }}>
          Our AI will craft a personalized, professional cover letter tailored specifically to this job description and your experience.
        </p>
        <button
          className="btn btn-primary btn-lg"
          onClick={() => generateCL()}
          disabled={isGenerating}
          id="generate-cover-letter-btn"
        >
          {isGenerating
            ? <><Spinner size="sm" /> Triggering generation…</>
            : <><Sparkles className="w-5 h-5" /> Generate Cover Letter</>
          }
        </button>
      </div>
    );
  }

  // Triggered — polling for result
  if (isClLoading || !clData) {
    return (
      <div className="card-elevated text-center py-10 animate-fade-in">
        <Spinner size="xl" className="mx-auto mb-4" />
        <p className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>
          Generating your cover letter…
        </p>
        <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
          This takes 20–60 seconds for the AI model to write.
        </p>
      </div>
    );
  }

  if (clData.status === 'IN_PROGRESS' || clData.status === 'PENDING') {
    return (
      <div className="card-elevated text-center py-10 animate-fade-in">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Spinner size="lg" />
          <div className="text-left">
            <p className="font-semibold" style={{ color: 'var(--color-text-primary)' }}>
              AI is writing your cover letter…
            </p>
            <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
              Status: {clData.status} · This may take a minute
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (clData.status === 'FAILURE') {
    return (
      <div className="card-elevated py-8 text-center animate-fade-in">
        <AlertCircle className="w-10 h-10 mx-auto mb-3" style={{ color: '#f43f5e' }} />
        <h3 className="font-bold mb-1" style={{ color: 'var(--color-text-primary)' }}>
          Cover Letter Generation Failed
        </h3>
        <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
          {clData.error_message || 'An error occurred. The LLM model may be unavailable.'}
        </p>
        <button
          className="btn btn-secondary btn-sm mt-4"
          onClick={() => { setTriggered(false); }}
        >
          <RefreshCw className="w-4 h-4" /> Try Again
        </button>
      </div>
    );
  }

  if (clData.status === 'SUCCESS' && clData.generated_text) {
    return (
      <CoverLetterEditor
        coverLetterText={clData.generated_text}
        jobTitle={parsedJd?.job_title}
        company={parsedJd?.company}
      />
    );
  }

  return null;
}

// ── Main Dashboard ─────────────────────────────────────────────────────────
export default function DashboardPage() {
  const { jobId } = useParams();

  const { data: job, isLoading, isError } = useJobPolling(jobId);

  if (isLoading) {
    return (
      <div className="page-container py-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1"><SkeletonCard height={340} /></div>
          <div className="lg:col-span-2 space-y-5">
            <SkeletonCard height={220} />
            <SkeletonCard height={280} />
          </div>
        </div>
      </div>
    );
  }

  if (isError || !job) {
    return (
      <div className="page-container py-20 text-center">
        <AlertCircle className="w-12 h-12 mx-auto mb-4" style={{ color: '#f43f5e' }} />
        <h2 className="text-xl font-bold mb-2" style={{ color: 'var(--color-text-primary)' }}>
          Could not load this analysis
        </h2>
        <p className="text-sm mb-6" style={{ color: 'var(--color-text-muted)' }}>
          Job ID: <code className="text-xs">{jobId}</code> — it may not exist or you may not have access.
        </p>
        <Link to="/" className="btn btn-primary">
          <ArrowLeft className="w-4 h-4" /> Start a New Analysis
        </Link>
      </div>
    );
  }

  const isFailed = job.status === 'FAILURE';
  const isSuccess = job.status === 'SUCCESS';
  const isProcessing = ['PENDING', 'IN_PROGRESS'].includes(job.status);

  return (
    <div className="hero-gradient-bg min-h-screen">
      <div className="page-container py-8">

        {/* ── Top bar ──────────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between mb-6 animate-fade-in">
          <div className="flex items-center gap-3">
            <Link
              to="/"
              className="btn btn-ghost btn-sm"
              id="dashboard-back-btn"
            >
              <ArrowLeft className="w-4 h-4" /> New Analysis
            </Link>
            <div className="h-4 w-px" style={{ background: 'var(--color-border-subtle)' }} />
            <StatusBadge status={job.status} />
          </div>

          {/* Job meta */}
          {job.parsed_jd && (
            <div className="hidden sm:flex items-center gap-4 text-sm"
              style={{ color: 'var(--color-text-muted)' }}
            >
              {job.parsed_jd.company && (
                <span className="flex items-center gap-1.5">
                  <Building2 className="w-3.5 h-3.5" />
                  {job.parsed_jd.company}
                </span>
              )}
              {job.parsed_jd.job_title && (
                <span className="flex items-center gap-1.5">
                  <Briefcase className="w-3.5 h-3.5" />
                  {job.parsed_jd.job_title}
                </span>
              )}
            </div>
          )}

          <div className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-muted)' }}>
            <Clock className="w-3.5 h-3.5" />
            {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
          </div>
        </div>

        {/* ── PROCESSING VIEW ──────────────────────────────────────────────── */}
        {isProcessing && (
          <div className="max-w-xl mx-auto">
            <ProcessingSteps status={job.status} />
          </div>
        )}

        {/* ── FAILURE VIEW ─────────────────────────────────────────────────── */}
        {isFailed && (
          <div className="max-w-xl mx-auto animate-scale-in">
            <div className="card-elevated text-center py-10 px-6">
              <AlertCircle className="w-14 h-14 mx-auto mb-4" style={{ color: '#f43f5e' }} />
              <h2 className="text-xl font-bold mb-3" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-display)' }}>
                Analysis Failed
              </h2>
              <p className="text-sm mb-6" style={{ color: 'var(--color-text-muted)' }}>
                {job.error_message || 'An unexpected error occurred during processing.'}
              </p>
              
              {/* Fallback copy-paste form for JD */}
              <div className="mt-4 p-5 bg-gray-50/5 rounded-xl border border-gray-200/20 text-left">
                <h3 className="font-semibold text-sm mb-2" style={{ color: 'var(--color-text-primary)'}}>Did URL scraping fail?</h3>
                <p className="text-xs mb-4" style={{ color: 'var(--color-text-muted)' }}>Copy and paste the job description text manually below to retry without creating a new job.</p>
                <JDRetryForm jobId={jobId} />
              </div>

              <div className="mt-6 flex items-center justify-center gap-3">
                <Link to="/" className="btn btn-secondary btn-sm">
                  <RefreshCw className="w-4 h-4" /> Start Over
                </Link>
              </div>
            </div>
          </div>
        )}

        {/* ── SUCCESS VIEW ─────────────────────────────────────────────────── */}
        {isSuccess && (
          <>
            {/* Main 3-column grid layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">

              {/* Left col: ATS Score */}
              <div className="lg:col-span-1">
                <ATSScoreCard
                  score={job.ats_score || 0}
                  breakdown={job.scoring_breakdown}
                />
              </div>

              {/* Right col: Keyword match + Suggestions */}
              <div className="lg:col-span-2 space-y-5">
                <KeywordMatchPanel
                  matchedSkills={job.scoring_breakdown?.matched_skills || []}
                  missingSkills={job.scoring_breakdown?.missing_skills || []}
                  resumeSkills={job.parsed_resume?.skills || []}
                />
                <SuggestionsPanel suggestions={job.suggestions || []} />
              </div>
            </div>

            {/* Cover letter full-width section */}
            <CoverLetterSection
              jobId={jobId}
              parsedJd={job.parsed_jd}
            />
          </>
        )}
      </div>
    </div>
  );
}
