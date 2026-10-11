/**
 * KeywordMatchPanel – Shows matched vs missing skills side by side.
 * Matched skills are highlighted green, missing are highlighted red.
 */

import { CheckCircle2, XCircle, Layers } from 'lucide-react';

function SkillChip({ skill, variant }) {
  const styles = {
    matched: {
      bg: 'rgba(16,185,129,0.12)',
      border: 'rgba(16,185,129,0.25)',
      color: '#10b981',
      icon: <CheckCircle2 className="w-3 h-3" />,
    },
    missing: {
      bg: 'rgba(244,63,94,0.1)',
      border: 'rgba(244,63,94,0.2)',
      color: '#f43f5e',
      icon: <XCircle className="w-3 h-3" />,
    },
  };

  const s = styles[variant];

  return (
    <span
      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium transition-transform hover:scale-105"
      style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}
    >
      {s.icon}
      {skill}
    </span>
  );
}

export default function KeywordMatchPanel({ matchedSkills = [], missingSkills = [], resumeSkills = [] }) {
  return (
    <div className="card-elevated animate-fade-in-up">
      <div className="flex items-center gap-2 mb-5">
        <div className="w-8 h-8 rounded-lg gradient-brand flex items-center justify-center">
          <Layers className="w-4 h-4 text-white" />
        </div>
        <div>
          <h2 className="font-bold text-base" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-display)' }}>
            Keyword Analysis
          </h2>
          <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Skills found vs skills missing in your resume
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Matched skills */}
        <div className="rounded-xl p-4" style={{ background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.15)' }}>
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle2 className="w-4 h-4" style={{ color: '#10b981' }} />
            <span className="text-sm font-semibold" style={{ color: '#10b981' }}>
              Matched ({matchedSkills.length})
            </span>
          </div>
          {matchedSkills.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {matchedSkills.map((skill) => (
                <SkillChip key={skill} skill={skill} variant="matched" />
              ))}
            </div>
          ) : (
            <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              No matching skills found yet.
            </p>
          )}
        </div>

        {/* Missing skills */}
        <div className="rounded-xl p-4" style={{ background: 'rgba(244,63,94,0.06)', border: '1px solid rgba(244,63,94,0.15)' }}>
          <div className="flex items-center gap-2 mb-3">
            <XCircle className="w-4 h-4" style={{ color: '#f43f5e' }} />
            <span className="text-sm font-semibold" style={{ color: '#f43f5e' }}>
              Missing ({missingSkills.length})
            </span>
          </div>
          {missingSkills.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {missingSkills.map((skill) => (
                <SkillChip key={skill} skill={skill} variant="missing" />
              ))}
            </div>
          ) : (
            <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              🎉 Your resume covers all required skills!
            </p>
          )}
        </div>
      </div>

      {/* Your skills section */}
      {resumeSkills.length > 0 && (
        <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--color-border-subtle)' }}>
          <p className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: 'var(--color-text-muted)' }}>
            All skills on your resume
          </p>
          <div className="flex flex-wrap gap-1.5">
            {resumeSkills.slice(0, 24).map((skill) => (
              <span
                key={skill}
                className="px-2 py-0.5 rounded-md text-xs font-medium"
                style={{ background: 'rgba(99,102,241,0.1)', color: 'var(--color-accent-indigo)', border: '1px solid var(--color-border-subtle)' }}
              >
                {skill}
              </span>
            ))}
            {resumeSkills.length > 24 && (
              <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                +{resumeSkills.length - 24} more
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
