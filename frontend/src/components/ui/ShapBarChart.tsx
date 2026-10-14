/**
 * CardioVision AI — SHAP Waterfall Bar Chart
 * =============================================
 * Renders a horizontal bar chart of SHAP feature contributions using Recharts.
 * Positive values (risk-increasing) are shown in red/rose.
 * Negative values (risk-decreasing) are shown in emerald/green.
 *
 * Labeled as "Model Feature Contribution" per explainability best practices.
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import type { FeatureContribution } from '@/types';

interface ShapBarChartProps {
  contributions: FeatureContribution[];
  topN?: number;
  title?: string;
  className?: string;
}

import { getFeatureCategory, CATEGORY_STYLES, MODEL_CONTRIBUTION_LABEL } from '@/utils';

interface TooltipPayload {
  value: number;
  payload: {
    feature: string;
    patient_value: string | number | boolean;
    shap_value: number;
    absolute_contribution: number;
    feature_category?: string;
  };
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  const isPositive = d.shap_value > 0;
  const cat = (d.feature_category || getFeatureCategory(d.feature)) as keyof typeof CATEGORY_STYLES;
  const catStyle = CATEGORY_STYLES[cat] || CATEGORY_STYLES.Clinical;

  return (
    <div className="bg-surface-card/95 backdrop-blur-md border border-surface-border rounded-xl p-3 text-xs shadow-card min-w-[240px]">
      <div className="flex items-center justify-between gap-2 mb-2 pb-1.5 border-b border-surface-border/60">
        <p className="font-semibold text-white font-display text-sm">{d.feature}</p>
        <span className={`px-2 py-0.5 rounded text-[10px] font-mono border font-semibold ${catStyle.badge}`}>
          {cat}
        </span>
      </div>
      <div className="space-y-1.5 font-mono text-gray-400">
        <div className="flex justify-between gap-4">
          <span>Patient Value:</span>
          <span className="text-gray-100 font-semibold">{String(d.patient_value)}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span>SHAP Value:</span>
          <span className={isPositive ? 'text-red-400 font-bold' : 'text-emerald-400 font-bold'}>
            {isPositive ? '+' : ''}{d.shap_value.toFixed(5)}
          </span>
        </div>
        <div className="flex justify-between gap-4">
          <span>Absolute Impact |SHAP|:</span>
          <span className="text-gray-200">{d.absolute_contribution.toFixed(5)}</span>
        </div>
        <div className="pt-1.5 mt-1 border-t border-surface-border/40 text-[10px] text-gray-400 font-sans">
          <span>Scope: {MODEL_CONTRIBUTION_LABEL}</span>
        </div>
      </div>
    </div>
  );
}

export function ShapBarChart({
  contributions,
  topN = 15,
  title = 'Top Feature Contributions',
  className = '',
}: ShapBarChartProps) {
  // Sort by absolute SHAP value, take topN
  const sorted = [...contributions]
    .sort((a, b) => b.absolute_contribution - a.absolute_contribution)
    .slice(0, topN);

  // Recharts wants data sorted so the chart reads top-to-bottom correctly for horizontal bars
  const chartData = sorted.reverse().map((item) => ({
    feature: item.feature.length > 22 ? item.feature.slice(0, 21) + '…' : item.feature,
    featureFull: item.feature,
    shap_value: parseFloat(item.shap_value.toFixed(5)),
    absolute_contribution: item.absolute_contribution,
    patient_value: item.patient_value,
    feature_category: item.feature_category,
  }));

  const maxAbs = Math.max(...chartData.map((d) => Math.abs(d.shap_value)), 0.001);

  return (
    <div className={`space-y-2 ${className}`}>
      {title && (
        <div className="flex items-center gap-3 mb-3">
          <h4 className="text-sm font-display font-semibold text-white">{title}</h4>
          <div className="flex items-center gap-3 text-[11px] font-mono ml-auto">
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 rounded-sm bg-red-500/70 inline-block" />
              <span className="text-gray-400">Increases Risk</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 rounded-sm bg-emerald-500/70 inline-block" />
              <span className="text-gray-400">Decreases Risk</span>
            </span>
          </div>
        </div>
      )}

      <ResponsiveContainer width="100%" height={Math.max(chartData.length * 32, 200)}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 0, right: 60, left: 0, bottom: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(30,39,64,0.8)"
            horizontal={false}
          />
          <XAxis
            type="number"
            domain={[-maxAbs * 1.1, maxAbs * 1.1]}
            tickFormatter={(v: number) => v.toFixed(3)}
            tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
            axisLine={{ stroke: '#1e2740' }}
            tickLine={{ stroke: '#1e2740' }}
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={130}
            tick={{ fill: '#d1d5db', fontSize: 11, fontFamily: 'Inter, system-ui, sans-serif' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: 'rgba(99,102,241,0.06)' }}
          />
          <ReferenceLine x={0} stroke="rgba(99,102,241,0.4)" strokeWidth={1.5} />
          <Bar dataKey="shap_value" radius={[0, 4, 4, 0]} maxBarSize={20}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.shap_value > 0 ? '#f43f5e' : '#22c55e'}
                fillOpacity={0.75}
                stroke={entry.shap_value > 0 ? '#f43f5e' : '#22c55e'}
                strokeOpacity={0.3}
                strokeWidth={1}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="text-[10px] text-gray-500 font-mono text-center pt-1">
        SHAP Value → Positive values increase predicted risk probability; Negative values decrease it
      </p>
    </div>
  );
}
