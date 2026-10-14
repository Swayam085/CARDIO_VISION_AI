/**
 * CardioVision AI — Navbar Component
 * =====================================
 * Sticky top navigation with:
 * - Logo with animated heartbeat
 * - Desktop nav links with active underline indicator
 * - Mobile hamburger menu (slide-down drawer)
 * - CTA button
 */

import { useState, useEffect } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { Activity, Menu, X, ChevronRight } from 'lucide-react';
import { clsx } from 'clsx';
import { APP_ROUTES } from '@/types';

const NAV_ROUTES = APP_ROUTES.slice(1); // Skip '/' home route

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  // Close mobile menu on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  // Prevent body scroll when mobile menu open
  useEffect(() => {
    document.body.style.overflow = mobileOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [mobileOpen]);

  return (
    <>
      <header
        id="main-navbar"
        className="sticky top-0 z-50 border-b border-surface-border"
        style={{ background: 'rgba(10, 11, 20, 0.90)', backdropFilter: 'blur(20px)' }}
      >
        <div className="section-container">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link
              to="/"
              id="logo-link"
              className="flex items-center gap-2.5 group"
              aria-label="CardioVision AI Home"
            >
              <div className="w-8 h-8 rounded-lg bg-primary-600 flex items-center justify-center
                              group-hover:bg-primary-500 transition-all duration-200 shadow-glow-primary">
                <Activity className="w-4 h-4 text-white animate-heartbeat" aria-hidden="true" />
              </div>
              <div className="flex flex-col leading-tight">
                <span className="text-sm font-bold font-display gradient-text-primary">
                  CardioVision AI
                </span>
                <span className="text-[10px] text-gray-500 font-mono tracking-widest uppercase">
                  Hackathon 2026
                </span>
              </div>
            </Link>

            {/* Desktop Nav */}
            <nav aria-label="Main navigation" className="hidden md:flex items-center gap-1">
              {NAV_ROUTES.map((route) => (
                <NavLink
                  key={route.path}
                  to={route.path}
                  id={`nav-${route.label.toLowerCase()}`}
                  title={route.description}
                  className={({ isActive }) =>
                    clsx('nav-link', isActive && 'active')
                  }
                >
                  {route.label}
                </NavLink>
              ))}
            </nav>

            {/* Desktop CTA + Mobile Toggle */}
            <div className="flex items-center gap-3">
              <Link
                to="/analyze"
                id="cta-analyze"
                className="hidden sm:inline-flex btn-accent text-xs px-4 py-2"
                aria-label="Start cardiovascular analysis"
              >
                <Activity className="w-3.5 h-3.5" aria-hidden="true" />
                Analyze
              </Link>

              {/* Mobile hamburger */}
              <button
                id="mobile-menu-toggle"
                onClick={() => setMobileOpen(!mobileOpen)}
                className="md:hidden w-9 h-9 flex items-center justify-center rounded-lg
                           border border-surface-border text-gray-400 hover:text-white
                           hover:border-primary-500 transition-all"
                aria-label={mobileOpen ? 'Close navigation menu' : 'Open navigation menu'}
                aria-expanded={mobileOpen}
              >
                {mobileOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Drawer */}
      {mobileOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm md:hidden"
            onClick={() => setMobileOpen(false)}
            aria-hidden="true"
          />

          {/* Drawer */}
          <div
            id="mobile-nav-drawer"
            className="fixed top-16 left-0 right-0 z-50 md:hidden
                       border-b border-surface-border animate-slide-up"
            style={{ background: 'rgba(10, 11, 20, 0.98)' }}
            role="navigation"
            aria-label="Mobile navigation"
          >
            <div className="section-container py-4 space-y-1">
              {NAV_ROUTES.map((route) => (
                <NavLink
                  key={route.path}
                  to={route.path}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center justify-between px-4 py-3 rounded-xl text-sm font-medium transition-all',
                      isActive
                        ? 'bg-primary-500/15 border border-primary-500/30 text-white'
                        : 'text-gray-400 hover:bg-surface-card hover:text-white border border-transparent'
                    )
                  }
                >
                  <span>{route.label}</span>
                  <ChevronRight className="w-4 h-4 opacity-40" />
                </NavLink>
              ))}

              <div className="pt-2 pb-1">
                <Link
                  to="/analyze"
                  className="btn-accent w-full justify-center"
                >
                  <Activity className="w-4 h-4" />
                  Start Analysis
                </Link>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}
