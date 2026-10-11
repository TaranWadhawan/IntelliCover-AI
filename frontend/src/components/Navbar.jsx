/**
 * Navbar – Top navigation bar.
 * Shows logo + auth state (logged in → avatar/logout; logged out → sign in link).
 */

import { Link, useNavigate } from 'react-router-dom';
import { Sparkles, LogOut, User } from 'lucide-react';
import useAuthStore from '../store/authStore';

export default function Navbar() {
  const { user, logout, isAuthenticated } = useAuthStore();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav
      className="glass sticky top-0 z-50 border-b"
      style={{ borderColor: 'var(--color-border-subtle)' }}
    >
      <div className="page-container">
        <div className="flex items-center justify-between h-16">

          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center gradient-brand shadow-lg"
              style={{ boxShadow: 'var(--shadow-glow-sm)' }}
            >
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span
              className="font-bold text-lg tracking-tight"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--color-text-primary)' }}
            >
              Intelli<span className="gradient-text">Cover</span>
            </span>
          </Link>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {isAuthenticated() && user ? (
              <>
                <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg"
                  style={{ background: 'rgba(99,102,241,0.08)', border: '1px solid var(--color-border-subtle)' }}
                >
                  <div className="w-5 h-5 rounded-full gradient-brand flex items-center justify-center">
                    <User className="w-3 h-3 text-white" />
                  </div>
                  <span className="text-sm font-medium" style={{ color: 'var(--color-text-secondary)' }}>
                    {user.full_name || user.email.split('@')[0]}
                  </span>
                </div>
                <button
                  onClick={handleLogout}
                  className="btn btn-ghost btn-sm flex items-center gap-1.5"
                  id="navbar-logout-btn"
                >
                  <LogOut className="w-4 h-4" />
                  <span className="hidden sm:inline">Logout</span>
                </button>
              </>
            ) : (
              <>
                <Link to="/login" className="btn btn-ghost btn-sm" id="navbar-login-link">
                  Sign In
                </Link>
                <Link to="/register" className="btn btn-primary btn-sm" id="navbar-register-link">
                  Get Started
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
