/**
 * CardioVision AI — 404 Not Found Page
 */

import { Link } from 'react-router-dom';
import { Activity } from 'lucide-react';

export function NotFoundPage() {
  return (
    <main id="not-found-page" className="section-container py-24 text-center animate-fade-in">
      <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl
                      bg-gray-800 border border-surface-border mb-6">
        <Activity className="w-8 h-8 text-gray-600" aria-hidden="true" />
      </div>
      <h1 className="text-6xl font-display font-black text-gray-700 mb-4">404</h1>
      <p className="text-xl text-gray-400 font-display font-semibold mb-2">Page not found</p>
      <p className="text-gray-600 text-sm mb-8">
        The route you navigated to does not exist.
      </p>
      <Link to="/" id="back-home-link" className="btn-primary">
        ← Back to Home
      </Link>
    </main>
  );
}
