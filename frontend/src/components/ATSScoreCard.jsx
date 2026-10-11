/**
 * ATSScoreCard – Animated circular gauge with score + breakdown.
 * The ring animates from 0 to the final score when it mounts.
 */

import { useEffect, useState } from 'react';
import { TrendingUp, Award, Zap, BookOpen } from 'lucide-react';

const RADIUS = 62;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function getScoreColor(score) {
  if (score >= 75) return '#10b981'; // emerald
  if (score >= 50) return '#f59e0b'; // amber
  return '#f43f5e';                  // rose
}

function getScoreLabel(score) {
  if (score >= 80) return 'Excellent Match';
  if (score >= 65) return 'Good Match';
  if (score >= 45) return 'Moderate Match';
  return 'Needs Improvement';
}

function ScoreBreakdownItem({ icon: Icon, label, value, color }) {
  return (
    <div className="flex items-center justify-between py-2.5"
      style={{ borderBottom: '1px solid var(--color-border-subtle)' }}
    >
      <div className="flex items-center gap-2.5">
        <div className="w-7 h-7 rounded-lg flex items-center justify-center"
          style={{ background: `${color}18`, border: `1px solid ${color}30` }}
        >
          <Icon className="w-4 h-4" style={{ color }} />
        </div>
        <span className="text-sm font-medium" style={{ color: 'var(--color-text-secondary)' }}>
          {label}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-24 h-1.5 rounded-full overflow-hidden"
          style={{ background: 'var(--color-bg-elevated)' }}
        >
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${value}%`, background: color }}
          />
        </div>
        <span className="text-sm font-bold w-10 text-right" style={{ color }}>
          {Math.round(value)}%
        </span>
      </div>
    </div>
  );
}

export default function ATSScoreCard({ score, breakdown }) {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(score);
    }, 200);
    return () => clearTimeout(timer);
  }, [score]);

  const color = getScoreColor(score);
  const offset = CIRCUMFERENCE - (animatedScore / 100) * CIRCUMFERENCE;

  return (
    <div className="card-elevated animate-scale-in">
      {/* Header */}
      <div className="flex items-center gap-2 mb-6">
        <div className="w-8 h-8 rounded-lg gradient-brand flex items-center justify-center">
          <Award className="w-4 h-4 text-white" />
        </div>
        <div>
          <h2 className="font-bold text-base" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-display)' }}>
            ATS Alignment Score
          </h2>
          <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            How well your resume matches this job
          </p>
        </div>
      </div>

      {/* Circular gauge */}
      <div className="flex flex-col items-center mb-6">
        <div className="relative animate-scale-in">
          <svg width="160" height="160" className="-rotate-90">
            {/* Track */}
            <circle
              cx="80" cy="80" r={RADIUS}
              fill="none"
              stroke="rgba(99,102,241,0.1)"
              strokeWidth="10"
            />
            {/* Progress */}
            <circle
              cx="80" cy="80" r={RADIUS}
              fill="none"
              stroke={color}
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={offset}
              style={{ transition: 'stroke-dashoffset 1.4s cubic-bezier(0.34, 1.56, 0.64, 1)', filter: `drop-shadow(0 0 8px ${color}60)` }}
            />
          </svg>
          {/* Score number */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-black" style={{ color, fontFamily: 'var(--font-display)' }}>
              {Math.round(animatedScore)}
            </span>
            <span className="text-xs font-semibold" style={{ color: 'var(--color-text-muted)' }}>out of 100</span>
          </div>
        </div>

        <div className="mt-2">
          <span className="badge" style={{ background: `${color}18`, color, border: `1px solid ${color}30`, fontSize: '0.8125rem', padding: '0.375rem 1rem' }}>
            {getScoreLabel(score)}
          </span>
        </div>
      </div>

      {/* Breakdown */}
      {breakdown && (
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest mb-3"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Score Breakdown
          </p>
          <div>
            <ScoreBreakdownItem
              icon={Zap}
              label="Skill Match (50%)"
              value={breakdown.hard_skill_score}
              color="#6366f1"
            />
            <ScoreBreakdownItem
              icon={TrendingUp}
              label="Semantic Fit (40%)"
              value={breakdown.semantic_score}
              color="#8b5cf6"
            />
            <ScoreBreakdownItem
              icon={BookOpen}
              label="Education (10%)"
              value={breakdown.education_score}
              color="#06b6d4"
            />
          </div>
        </div>
      )}
    </div>
  );
}
