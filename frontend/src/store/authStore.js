/**
 * Zustand Global Auth Store
 * 
 * Persists JWT token in localStorage.
 * Exposes: user, token, login(), logout(), setUser()
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authApi } from '../api/client';

const useAuthStore = create(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,

      /** Called after successful login. */
      login: async (email, password) => {
        set({ isLoading: true });
        try {
          const { data } = await authApi.login({ email, password });
          localStorage.setItem('intellicover_token', data.access_token);
          set({ token: data.access_token, isLoading: false });

          // Fetch the user profile immediately
          const { data: user } = await authApi.me();
          set({ user });
          return { success: true };
        } catch (err) {
          set({ isLoading: false });
          const message = err.response?.data?.detail || 'Login failed.';
          return { success: false, message };
        }
      },

      /** Called after successful registration. */
      register: async (email, password, fullName) => {
        set({ isLoading: true });
        try {
          await authApi.register({ email, password, full_name: fullName });
          set({ isLoading: false });
          return { success: true };
        } catch (err) {
          set({ isLoading: false });
          const message = err.response?.data?.detail || 'Registration failed.';
          return { success: false, message };
        }
      },

      logout: () => {
        localStorage.removeItem('intellicover_token');
        set({ user: null, token: null });
      },

      setUser: (user) => set({ user }),
      isAuthenticated: () => !!get().token,
    }),
    {
      name: 'intellicover-auth',
      partialize: (state) => ({ token: state.token, user: state.user }),
    }
  )
);

export default useAuthStore;
