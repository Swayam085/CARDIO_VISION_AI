/**
 * CardioVision AI — Home Page (/)
 * =================================
 * Landing page with hero, feature cards, vessel overview, and safety disclaimer.
 */

import { Link } from 'react-router-dom';
import { Activity, Brain, Eye, FileText, Zap, Shield, Database } from 'lucide-react';
import { SafetyDisclaimer } from '@/components/ui/SafetyDisclaimer';
import { BackendStatus } from '@/components/ui/BackendStatus';

const VESSELS = [
  { id: 'LAD', name: 'Left Anterior Descending', abbr: 'LAD', color: '#f43f5e', glow: 'shadow-glow-lad', desc: 'Supplies blood to the front of the left heart.' },
  { id: 'LCX', name: 'Left Circumflex',          abbr: 'LCX', color: '#f97316', glow: 'shadow-glow-lcx', desc: 'Supplies blood to the left heart wall.' },
  { id: 'RCA', name: 'Right Coronary Artery',    abbr: 'RCA', color: '#3b82f6', glow: 'shadow-glow-rca', desc: 'Supplies blood to the right heart and inferior wall.' },
];

const FEATURES = [
  {
    icon: Brain,
    title: 'Multi-Vessel Prediction',
    desc: 'XGBoost/LightGBM models predict stenosis risk for LAD, LCX, and RCA from clinical features.',
    color: 'text-primary-400',
    bg:    'bg-primary-500/10',
  },
  {
    icon: Eye,
    title: '3D Risk Visualization',
    desc: 'Model-predicted vessel-level risk mapped to 3D coronary anatomy. BodyParts3D anatomical reference.',
    color: 'text-accent-400',
    bg:    'bg-accent-500/10',
  },
  {
    icon: Zap,
    title: 'SHAP Explainability',
    desc: 'Every prediction is explained with SHAP values — which clinical features drove the risk score.',
    color: 'text-yellow-400',
    bg:    'bg-yellow-500/10',
  },
  {
    icon: Database,
    title: 'UCI Dataset (ID 411)',
    desc: 'Trained on the Z-Alizadeh Sani Extension Dataset. 303 patients, 59 clinical features. No data leakage.',
    color: 'text-emerald-400',
    bg:    'bg-emerald-500/10',
  },
  {
    icon: Shield,
    title: 'Leakage-Free Training',
    desc: 'Formal audit ensures LAD, LCX, RCA, Cath never appear as model inputs. Targets stay targets.',
    color: 'text-cyan-400',
    bg:    'bg-cyan-500/10',
  },
  {
    icon: FileText,
    title: 'Clinical Reports',
    desc: 'Generate a comprehensive clinical summary with risk breakdown, feature importance, and SHAP charts.',
    color: 'text-violet-400',
    bg:    'bg-violet-500/10',
  },
];

export function HomePage() {
  return (
    <main id="home-page" className="relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-grid-pattern pointer-events-none" aria-hidden="true" />
      <div className="absolute inset-0 bg-glow-primary pointer-events-none" aria-hidden="true" />

      {/* ── Hero ────────────────────────────────────────────── */}
      <section id="hero" className="relative section-container pt-20 pb-16 text-center">
        {/* Heartbeat icon */}
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 rounded-2xl bg-primary-600/20 border border-primary-500/30
                          flex items-center justify-center shadow-glow-primary animate-float">
            <Activity className="w-8 h-8 text-primary-400 animate-heartbeat" aria-hidden="true" />
          </div>
        </div>

        {/* Hackathon badge */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
                        bg-primary-500/10 border border-primary-500/30 text-primary-300 text-xs
                        font-mono uppercase tracking-wider mb-6">
          <Zap className="w-3 h-3" aria-hidden="true" />
          Multimodal AI Hackathon 2026 — Cardiovascular Track
        </div>

        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-display font-black text-balance mb-4">
          <span className="gradient-text-primary">CardioVision</span>
          <span className="text-white"> AI</span>
        </h1>

        <p className="text-lg sm:text-xl text-gray-400 font-light max-w-2xl mx-auto mb-3 text-balance">
          AI-Powered Cardiovascular Risk Visualization &amp;{' '}
          <span className="text-accent-400 font-medium">Multi-Vessel Stenosis Prediction</span>
        </p>

        <p className="text-sm text-gray-600 font-mono mb-10">
          LAD · LCX · RCA · UCI Dataset ID 411 · XGBoost · SHAP · Three.js
        </p>

        <div className="flex flex-wrap justify-center gap-4 mb-10">
          <Link to="/analyze" id="hero-cta-analyze" className="btn-accent">
            <Activity className="w-4 h-4" aria-hidden="true" />
            Start Analysis
          </Link>
          <Link to="/documentation" id="hero-cta-docs" className="btn-outline">
            <Database className="w-4 h-4" aria-hidden="true" />
            Dataset &amp; Docs
          </Link>
        </div>

        {/* Backend status */}
        <div className="flex justify-center">
          <BackendStatus />
        </div>
      </section>

      {/* ── Vessel Overview ──────────────────────────────────── */}
      <section id="vessels" className="section-container pb-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {VESSELS.map((v) => (
            <div
              key={v.id}
              id={`vessel-card-${v.id.toLowerCase()}`}
              className="glass-card p-5 hover:scale-[1.02] transition-transform duration-200"
            >
              <div className="flex items-center gap-3 mb-3">
                <div
                  className={`w-10 h-10 rounded-lg flex items-center justify-center ${v.glow}`}
                  style={{ background: `${v.color}20`, border: `1px solid ${v.color}40` }}
                >
                  <Activity className="w-5 h-5" style={{ color: v.color }} aria-hidden="true" />
                </div>
                <div>
                  <span
                    className={`vessel-badge vessel-badge--${v.abbr.toLowerCase()}`}
                    aria-label={`${v.name} (${v.abbr})`}
                  >
                    {v.abbr}
                  </span>
                </div>
              </div>
              <h3 className="text-sm font-semibold text-white mb-1">{v.name}</h3>
              <p className="text-xs text-gray-500 leading-relaxed">{v.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Feature Grid ─────────────────────────────────────── */}
      <section id="features" className="section-container pb-16">
        <h2 className="text-2xl font-display font-bold text-center mb-10 gradient-text-primary">
          Technical Architecture
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f) => {
            const Icon = f.icon;
            return (
              <div
                key={f.title}
                id={`feature-${f.title.replace(/\s+/g, '-').toLowerCase()}`}
                className="glass-card p-5 hover:border-primary-800 transition-colors duration-200 animate-fade-in"
              >
                <div className={`w-9 h-9 rounded-lg ${f.bg} flex items-center justify-center mb-3`}>
                  <Icon className={`w-4.5 h-4.5 ${f.color}`} aria-hidden="true" />
                </div>
                <h3 className="text-sm font-semibold text-white mb-2">{f.title}</h3>
                <p className="text-xs text-gray-500 leading-relaxed">{f.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Safety Disclaimer ────────────────────────────────── */}
      <section id="safety-section" className="section-container pb-20">
        <SafetyDisclaimer showVizNote={true} className="max-w-3xl mx-auto" />
      </section>
    </main>
  );
}
