/**
 * Spinner – Animated loading indicator.
 */

export default function Spinner({ size = 'md', className = '' }) {
  const sizes = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-10 h-10 border-[3px]',
    xl: 'w-14 h-14 border-4',
  };

  return (
    <div
      className={`rounded-full animate-spin-slow ${sizes[size]} ${className}`}
      style={{
        borderColor: 'rgba(99,102,241,0.2)',
        borderTopColor: 'var(--color-accent-indigo)',
      }}
      role="status"
      aria-label="Loading"
    />
  );
}
