/**
 * LoginPage – JWT authentication form.
 */

import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Sparkles, Mail, Lock, Eye, EyeOff, ArrowRight } from 'lucide-react';
import useAuthStore from '../store/authStore';
import Spinner from '../components/Spinner';

export default function LoginPage() {
  const { login, isLoading } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();

  const from = location.state?.from?.pathname || '/';

  const [email, setEmail]         = useState('');
  const [password, setPassword]   = useState('');
  const [showPass, setShowPass]   = useState(false);
  const [error, setError]         = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    const result = await login(email, password);
    if (result.success) {
      navigate(from, { replace: true });
    } else {
      setError(result.message);
    }
  };

  return (
    <div className="hero-gradient-bg min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-md animate-scale-in">

        {/* Logo */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl gradient-brand flex items-center justify-center mx-auto mb-4"
            style={{ boxShadow: 'var(--shadow-glow-md)' }}
          >
            <Sparkles className="w-7 h-7 text-white" />
          </div>
          <h1 className="text-2xl font-black" style={{ fontFamily: 'var(--font-display)', color: 'var(--color-text-primary)' }}>
            Welcome back
          </h1>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
            Sign in to your IntelliCover account
          </p>
        </div>

        {/* Card */}
        <div className="card-elevated">
          <form onSubmit={handleSubmit} id="login-form">

            {/* Email */}
            <div className="mb-4">
              <label className="input-label" htmlFor="login-email">Email address</label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none"
                  style={{ color: 'var(--color-text-muted)' }}
                />
                <input
                  id="login-email"
                  type="email"
                  className="input"
                  style={{ paddingLeft: '2.5rem' }}
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                />
              </div>
            </div>

            {/* Password */}
            <div className="mb-6">
              <label className="input-label" htmlFor="login-password">Password</label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none"
                  style={{ color: 'var(--color-text-muted)' }}
                />
                <input
                  id="login-password"
                  type={showPass ? 'text' : 'password'}
                  className="input"
                  style={{ paddingLeft: '2.5rem', paddingRight: '2.75rem' }}
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  className="absolute right-3.5 top-1/2 -translate-y-1/2"
                  onClick={() => setShowPass(!showPass)}
                  aria-label="Toggle password visibility"
                >
                  {showPass
                    ? <EyeOff className="w-4 h-4" style={{ color: 'var(--color-text-muted)' }} />
                    : <Eye className="w-4 h-4" style={{ color: 'var(--color-text-muted)' }} />
                  }
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div
                className="mb-5 p-3 rounded-xl animate-fade-in text-sm"
                style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.2)', color: '#f43f5e' }}
              >
                ⚠️ {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              className="btn btn-primary btn-lg w-full"
              disabled={isLoading}
              id="login-submit-btn"
            >
              {isLoading
                ? <><Spinner size="sm" /> Signing in…</>
                : <><span>Sign In</span><ArrowRight className="w-5 h-5" /></>
              }
            </button>
          </form>

          <div className="divider" />

          <p className="text-center text-sm" style={{ color: 'var(--color-text-muted)' }}>
            Don't have an account?{' '}
            <Link to="/register" className="font-semibold" style={{ color: 'var(--color-accent-indigo)' }}
              id="login-register-link"
            >
              Create one for free
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
