/**
 * App.jsx – Root component.
 * Sets up React Router and wraps the app in TanStack Query + Zustand.
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import Navbar from './components/Navbar';
import ProtectedRoute from './components/ProtectedRoute';

import HomePage      from './pages/HomePage';
import LoginPage     from './pages/LoginPage';
import RegisterPage  from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col" style={{ background: 'var(--color-bg-base)' }}>
          <Navbar />

          <main className="flex-1">
            <Routes>
              {/* Public routes */}
              <Route path="/"         element={<HomePage />} />
              <Route path="/login"    element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />

              {/* Protected routes */}
              <Route
                path="/dashboard/:jobId"
                element={
                  <ProtectedRoute>
                    <DashboardPage />
                  </ProtectedRoute>
                }
              />

              {/* 404 */}
              <Route
                path="*"
                element={
                  <div className="page-container py-20 text-center">
                    <h1 className="text-6xl font-black gradient-text mb-4"
                      style={{ fontFamily: 'var(--font-display)' }}
                    >
                      404
                    </h1>
                    <p className="text-lg mb-6" style={{ color: 'var(--color-text-muted)' }}>
                      This page doesn't exist.
                    </p>
                    <a href="/" className="btn btn-primary">
                      Go Home
                    </a>
                  </div>
                }
              />
            </Routes>
          </main>

          {/* Footer */}
          <footer className="py-6 text-center"
            style={{ borderTop: '1px solid var(--color-border-subtle)' }}
          >
            <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              © 2025 IntelliCover AI · AI-generated content is a starting point, not a final answer.
            </p>
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
