/**
 * RegisterPage – New user registration form.
 */

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Sparkles, Mail, Lock, User, Eye, EyeOff, ArrowRight, CheckCircle2 } from 'lucide-react';
import useAuthStore from '../store/authStore';
import Spinner from '../components/Spinner';

export default function RegisterPage() {
  const { register, login, isLoading } = useAuthStore();
  const navigate = useNavigate();

  const [fullName, setFullName]   = useState('');
  const [email, setEmail]         = useState('');
  const [password, setPassword]   = useState('');
  const [showPass, setShowPass]   = useState(false);
  const [error, setError]         = useState('');
  const [success, setSuccess]     = useState(false);

  const passwordStrength = () => {
    if (password.length === 0) return 0;
    let score = 0;
    if (password.length >= 8)       score++;
    if (/[A-Z]/.test(password))     score++;
    if (/[0-9]/.test(password))     score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;
    return score;
  };

  const strength = passwordStrength();
  const strengthColors = ['', '#f43f5e', '#f59e0b', '#06b6d4', '#10b981'];
  const strengthLabels = ['', 'Weak', 'Fair', 'Good', 'Strong'];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (password.length < 8) {
      setError('Password must be at least 8 characters.');
      return;
    }

    const result = await register(email, password, fullName);
    if (result.success) {
      // Auto-login after registration
      const loginResult = await login(email, password);
      if (loginResult.success) {
        navigate('/');
      } else {
        setSuccess(true);
        setTimeout(() => navigate('/login'), 2000);
      }
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
            Create your account
          </h1>
          <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
            Free to start — no credit card needed
          </p>
        </div>

        {/* Card */}
        <div className="card-elevated">

          {success ? (
            <div className="text-center py-6 animate-scale-in">
              <CheckCircle2 className="w-12 h-12 mx-auto mb-3" style={{ color: '#10b981' }} />
              <h3 className="font-bold text-lg mb-1" style={{ color: 'var(--color-text-primary)' }}>
                Account Created!
              </h3>
              <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                Redirecting you to sign in…
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} id="register-form">

              {/* Full name */}
              <div className="mb-4">
                <label className="input-label" htmlFor="register-name">Full Name</label>
                <div className="relative">
                  <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none"
                    style={{ color: 'var(--color-text-muted)' }}
                  />
                  <input
                    id="register-name"
                    type="text"
                    className="input"
                    style={{ paddingLeft: '2.5rem' }}
                    placeholder="Jane Smith"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    autoComplete="name"
                  />
                </div>
              </div>

              {/* Email */}
              <div className="mb-4">
                <label className="input-label" htmlFor="register-email">Email address</label>
                <div className="relative">
                  <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none"
                    style={{ color: 'var(--color-text-muted)' }}
                  />
                  <input
                    id="register-email"
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
              <div className="mb-2">
                <label className="input-label" htmlFor="register-password">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none"
                    style={{ color: 'var(--color-text-muted)' }}
                  />
                  <input
                    id="register-password"
                    type={showPass ? 'text' : 'password'}
                    className="input"
                    style={{ paddingLeft: '2.5rem', paddingRight: '2.75rem' }}
                    placeholder="Min. 8 characters"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    autoComplete="new-password"
                  />
                  <button
                    type="button"
                    className="absolute right-3.5 top-1/2 -translate-y-1/2"
                    onClick={() => setShowPass(!showPass)}
                  >
                    {showPass
                      ? <EyeOff className="w-4 h-4" style={{ color: 'var(--color-text-muted)' }} />
                      : <Eye className="w-4 h-4" style={{ color: 'var(--color-text-muted)' }} />
                    }
                  </button>
                </div>
              </div>

              {/* Password strength */}
              {password.length > 0 && (
                <div className="mb-5 animate-fade-in">
                  <div className="flex gap-1 mb-1">
                    {[1,2,3,4].map((i) => (
                      <div
                        key={i}
                        className="h-1 flex-1 rounded-full transition-all duration-300"
                        style={{ background: i <= strength ? strengthColors[strength] : 'var(--color-bg-elevated)' }}
                      />
                    ))}
                  </div>
                  <p className="text-xs" style={{ color: strengthColors[strength] }}>
                    {strengthLabels[strength]}
                  </p>
                </div>
              )}

              {/* Error */}
              {error && (
                <div
                  className="mb-5 p-3 rounded-xl animate-fade-in text-sm"
                  style={{ background: 'rgba(244,63,94,0.08)', border: '1px solid rgba(244,63,94,0.2)', color: '#f43f5e' }}
                >
                  ⚠️ {error}
                </div>
              )}

              <button
                type="submit"
                className="btn btn-primary btn-lg w-full"
                disabled={isLoading}
                id="register-submit-btn"
              >
                {isLoading
                  ? <><Spinner size="sm" /> Creating account…</>
                  : <><span>Create Account</span><ArrowRight className="w-5 h-5" /></>
                }
              </button>
            </form>
          )}

          {!success && (
            <>
              <div className="divider" />
              <p className="text-center text-sm" style={{ color: 'var(--color-text-muted)' }}>
                Already have an account?{' '}
                <Link to="/login" className="font-semibold" style={{ color: 'var(--color-accent-indigo)' }}
                  id="register-login-link"
                >
                  Sign in
                </Link>
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
