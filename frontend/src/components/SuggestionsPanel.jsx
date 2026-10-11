/**
 * SuggestionsPanel – Displays AI-generated resume improvement suggestions.
 * Each suggestion fades in with a staggered delay for a cinematic reveal.
 */

import { Lightbulb, ChevronRight } from 'lucide-react';

export default function SuggestionsPanel({ suggestions = [] }) {
  return (
    <div className="card-elevated animate-fade-in-up delay-200">
      <div className="flex items-center gap-2 mb-5">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: 'rgba(245,158,11,0.15)', border: '1px solid rgba(245,158,11,0.25)' }}
        >
          <Lightbulb className="w-4 h-4" style={{ color: '#f59e0b' }} />
        </div>
        <div>
          <h2 className="font-bold text-base" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-display)' }}>
            AI Suggestions
          </h2>
          <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            Actionable improvements to boost your alignment
          </p>
        </div>
      </div>

      {suggestions.length === 0 ? (
        <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
          No suggestions generated yet.
        </p>
      ) : (
        <div className="space-y-3">
          {suggestions.map((suggestion, index) => (
            <div
              key={index}
              className="rounded-xl p-4 flex gap-3 animate-fade-in-up group transition-all hover:scale-[1.01]"
              style={{
                background: 'rgba(245,158,11,0.06)',
                border: '1px solid rgba(245,158,11,0.12)',
                animationDelay: `${index * 100}ms`,
                animationFillMode: 'both',
              }}
            >
              <div className="flex-shrink-0 mt-0.5">
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: 'rgba(245,158,11,0.2)', color: '#f59e0b' }}
                >
                  {index + 1}
                </div>
              </div>
              <div className="flex-1">
                <p className="text-sm leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                  {suggestion}
                </p>
              </div>
              <ChevronRight
                className="w-4 h-4 flex-shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ color: '#f59e0b' }}
              />
            </div>
          ))}
        </div>
      )}

      <div className="mt-5 flex items-start gap-2 p-3 rounded-lg"
        style={{ background: 'rgba(99,102,241,0.06)', border: '1px solid var(--color-border-subtle)' }}
      >
        <span className="text-base">ℹ️</span>
        <p className="text-xs leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
          <strong style={{ color: 'var(--color-text-secondary)' }}>Disclaimer:</strong> These suggestions are AI-generated. Please review them carefully and apply only the ones that genuinely reflect your experience.
        </p>
      </div>
    </div>
  );
}
