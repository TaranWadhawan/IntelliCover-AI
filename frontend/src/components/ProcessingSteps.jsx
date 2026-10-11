/**
 * ProcessingSteps – Real-time multi-step progress indicator.
 * Transforms the waiting period from passive to engaging.
 * 
 * Stages map to the Celery task pipeline stages.
 */

import { CheckCircle2, Loader2, Circle, AlertCircle } from 'lucide-react';

const STAGES = [
  { id: 'parsing',    label: 'Parsing Resume',           desc: 'Extracting structured data from your PDF' },
  { id: 'scraping',   label: 'Scraping Job Description', desc: 'Fetching content from the job URL' },
  { id: 'scoring',    label: 'Calculating ATS Score',    desc: 'Running semantic & skill analysis' },
  { id: 'generating', label: 'Generating Suggestions',   desc: 'AI career coach crafting your action plan' },
];

function getStageIndex(status) {
  switch (status) {
    case 'PENDING':     return -1;
    case 'IN_PROGRESS': return 1;  // start at "Scraping"
    case 'SUCCESS':     return 4;  // all done
    case 'FAILURE':     return -2; // error
    default:            return 0;
  }
}

function StageIcon({ stageIdx, currentIdx, isFailed }) {
  if (isFailed && stageIdx <= currentIdx) {
    return <AlertCircle className="w-5 h-5" style={{ color: '#f43f5e' }} />;
  }
  if (stageIdx < currentIdx) {
    return <CheckCircle2 className="w-5 h-5" style={{ color: '#10b981' }} />;
  }
  if (stageIdx === currentIdx) {
    return <Loader2 className="w-5 h-5 animate-spin-slow" style={{ color: '#6366f1' }} />;
  }
  return <Circle className="w-5 h-5" style={{ color: 'var(--color-text-muted)' }} />;
}

export default function ProcessingSteps({ status }) {
  const currentIdx = getStageIndex(status);
  const isFailed = status === 'FAILURE';
  const isPending = status === 'PENDING';

  return (
    <div className="card-elevated">
      <div className="flex items-center gap-3 mb-6">
        <div className="relative">
          <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center animate-pulse-glow">
            <Loader2 className="w-5 h-5 text-white animate-spin-slow" />
          </div>
        </div>
        <div>
          <h2 className="font-bold" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-display)' }}>
            {isFailed ? 'Analysis Failed' : isPending ? 'Queued for Processing' : 'Analyzing Your Application…'}
          </h2>
          <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
            {isFailed
              ? 'Something went wrong. Check the error details below.'
              : isPending
              ? 'Your job is in queue. This usually takes 15–30 seconds.'
              : 'Our AI is working through each stage below.'}
          </p>
        </div>
      </div>

      <div className="space-y-1">
        {STAGES.map((stage, idx) => {
          const isActive  = idx === currentIdx;
          const isDone    = idx < currentIdx;
          const isPending = idx > currentIdx;

          return (
            <div
              key={stage.id}
              className="flex items-start gap-3 p-3 rounded-xl transition-all duration-300"
              style={{
                background: isActive
                  ? 'rgba(99,102,241,0.08)'
                  : isDone
                  ? 'rgba(16,185,129,0.05)'
                  : 'transparent',
                border: isActive
                  ? '1px solid rgba(99,102,241,0.2)'
                  : isDone
                  ? '1px solid rgba(16,185,129,0.1)'
                  : '1px solid transparent',
              }}
            >
              <div className="mt-0.5 flex-shrink-0">
                <StageIcon stageIdx={idx} currentIdx={currentIdx} isFailed={isFailed} />
              </div>
              <div>
                <p
                  className="text-sm font-semibold"
                  style={{
                    color: isActive
                      ? 'var(--color-text-primary)'
                      : isDone
                      ? '#10b981'
                      : 'var(--color-text-muted)',
                  }}
                >
                  {stage.label}
                </p>
                {isActive && (
                  <p className="text-xs mt-0.5 animate-fade-in" style={{ color: 'var(--color-text-muted)' }}>
                    {stage.desc}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
