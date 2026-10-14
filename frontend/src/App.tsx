/**
 * CardioVision AI — App Router
 * =============================
 * Defines all application routes using React Router v7.
 *
 * Routes:
 *   /               → HomePage
 *   /analyze        → AnalyzePage
 *   /results        → ResultsPage
 *   /visualization  → VisualizationPage
 *   /insights       → InsightsPage
 *   /report         → ReportPage
 *   /documentation  → DocumentationPage
 *   *               → NotFoundPage
 */

import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Navbar } from '@/components/ui/Navbar';
import { SafetyDisclaimer } from '@/components/ui/SafetyDisclaimer';
import { AnalysisProvider } from '@/context/AnalysisContext';

// Lazy-loaded pages for optimal bundle splitting and fast initial page loads
const HomePage          = lazy(() => import('@/pages/HomePage').then((m) => ({ default: m.HomePage })));
const AnalyzePage       = lazy(() => import('@/pages/AnalyzePage').then((m) => ({ default: m.AnalyzePage })));
const ResultsPage       = lazy(() => import('@/pages/ResultsPage').then((m) => ({ default: m.ResultsPage })));
const VisualizationPage = lazy(() => import('@/pages/VisualizationPage').then((m) => ({ default: m.VisualizationPage })));
const InsightsPage      = lazy(() => import('@/pages/InsightsPage').then((m) => ({ default: m.InsightsPage })));
const MetricsPage       = lazy(() => import('@/pages/MetricsPage').then((m) => ({ default: m.MetricsPage })));
const ReportPage        = lazy(() => import('@/pages/ReportPage').then((m) => ({ default: m.ReportPage })));
const DocumentationPage = lazy(() => import('@/pages/DocumentationPage').then((m) => ({ default: m.DocumentationPage })));
const NotFoundPage      = lazy(() => import('@/pages/NotFoundPage').then((m) => ({ default: m.NotFoundPage })));

function PageLoading() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] gap-3">
      <div className="w-10 h-10 border-3 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
      <span className="text-xs text-gray-500 font-mono tracking-wider">LOADING CARDIOVISION...</span>
    </div>
  );
}

function Footer() {
  return (
    <footer id="app-footer" className="border-t border-surface-border mt-auto">
      <div className="section-container py-6">
        <SafetyDisclaimer compact className="mb-4" />
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-gray-600">
          <span>CardioVision AI — Multimodal AI Hackathon 2026</span>
          <div className="flex items-center gap-4">
            <a
              href="https://archive.ics.uci.edu/dataset/411/extention+of+z"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary-400 transition-colors"
              id="footer-uci-link"
            >
              UCI Dataset ID 411
            </a>
            <a
              href="https://lifesciencedb.jp/bp3d"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary-400 transition-colors"
              id="footer-anatomy-link"
            >
              BodyParts3D / DBCLS
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}

function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div id="app-layout" className="min-h-screen flex flex-col bg-surface-dark">
      <Navbar />
      <div className="flex-1">{children}</div>
      <Footer />
    </div>
  );
}

export function App() {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <AppLayout>
          <Suspense fallback={<PageLoading />}>
            <Routes>
              <Route path="/"               element={<HomePage />} />
              <Route path="/analyze"        element={<AnalyzePage />} />
              <Route path="/results"        element={<ResultsPage />} />
              <Route path="/visualization"  element={<VisualizationPage />} />
              <Route path="/insights"       element={<InsightsPage />} />
              <Route path="/metrics"        element={<MetricsPage />} />
              <Route path="/report"         element={<ReportPage />} />
              <Route path="/documentation"  element={<DocumentationPage />} />
              <Route path="*"               element={<NotFoundPage />} />
            </Routes>
          </Suspense>
        </AppLayout>
      </BrowserRouter>
    </AnalysisProvider>
  );
}
