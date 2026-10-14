/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // CardioVision AI design system
        primary: {
          50:  '#f0f4ff',
          100: '#e0e9ff',
          200: '#c7d6ff',
          300: '#a5b9ff',
          400: '#8193ff',
          500: '#6366f1',  // indigo-500
          600: '#4f46e5',
          700: '#4338ca',
          800: '#3730a3',
          900: '#312e81',
          950: '#1e1b4b',
        },
        accent: {
          50:  '#fff1f2',
          100: '#ffe4e6',
          200: '#fecdd3',
          300: '#fda4af',
          400: '#fb7185',
          500: '#f43f5e',  // rose-500 — cardiac red
          600: '#e11d48',
          700: '#be123c',
          800: '#9f1239',
          900: '#881337',
        },
        vessel: {
          lad: '#f43f5e',   // rose — LAD
          lcx: '#f97316',   // orange — LCX
          rca: '#3b82f6',   // blue — RCA
          healthy: '#22c55e', // green
          warning: '#f59e0b',
          critical: '#ef4444',
        },
        surface: {
          dark:    '#0a0b14',
          darker:  '#060710',
          card:    '#111827',
          cardAlt: '#1a2030',
          border:  '#1e2740',
        },
      },
      fontFamily: {
        sans:  ['Inter', 'system-ui', 'sans-serif'],
        mono:  ['JetBrains Mono', 'monospace'],
        display: ['Outfit', 'sans-serif'],
      },
      backgroundImage: {
        'gradient-radial':   'radial-gradient(var(--tw-gradient-stops))',
        'gradient-cardiac':  'linear-gradient(135deg, #0a0b14 0%, #1a0a1e 50%, #0a0b14 100%)',
        'glow-primary':      'radial-gradient(circle at 50% 50%, rgba(99,102,241,0.15) 0%, transparent 70%)',
        'glow-accent':       'radial-gradient(circle at 50% 50%, rgba(244,63,94,0.15) 0%, transparent 70%)',
      },
      boxShadow: {
        'glow-primary': '0 0 20px rgba(99,102,241,0.3)',
        'glow-accent':  '0 0 20px rgba(244,63,94,0.3)',
        'glow-lad':     '0 0 15px rgba(244,63,94,0.4)',
        'glow-lcx':     '0 0 15px rgba(249,115,22,0.4)',
        'glow-rca':     '0 0 15px rgba(59,130,246,0.4)',
        'card':         '0 4px 32px rgba(0,0,0,0.4)',
      },
      animation: {
        'pulse-slow':    'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'heartbeat':     'heartbeat 1.2s ease-in-out infinite',
        'slide-up':      'slideUp 0.4s ease-out',
        'fade-in':       'fadeIn 0.3s ease-out',
        'glow-pulse':    'glowPulse 2s ease-in-out infinite',
      },
      keyframes: {
        heartbeat: {
          '0%, 100%': { transform: 'scale(1)' },
          '14%':      { transform: 'scale(1.07)' },
          '28%':      { transform: 'scale(1)' },
          '42%':      { transform: 'scale(1.05)' },
          '70%':      { transform: 'scale(1)' },
        },
        slideUp: {
          '0%':   { opacity: '0', transform: 'translateY(16px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 10px rgba(99,102,241,0.2)' },
          '50%':      { boxShadow: '0 0 25px rgba(99,102,241,0.5)' },
        },
      },
    },
  },
  plugins: [],
}
