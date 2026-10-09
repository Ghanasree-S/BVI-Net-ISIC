import React from 'react';
import { PredictionMetrics, OrganType } from '../types';
import { ORGAN_CONFIGS } from '../data/organData';
import {
  Activity,
  Zap,
  Cpu,
  Target,
  BarChart3,
  CheckCircle2,
  TrendingUp,
  Sparkles,
  Layers,
} from 'lucide-react';

interface StatsPanelProps {
  metrics: PredictionMetrics;
  organType: OrganType;
}

export const StatsPanel: React.FC<StatsPanelProps> = ({ metrics, organType }) => {
  const currentOrgan = ORGAN_CONFIGS[organType];

  // Quality classification
  const getDiceQuality = (dice: number) => {
    if (dice >= 0.92) return { label: 'High Precision (Clinical Grade)', color: 'text-emerald-700 bg-emerald-50 border-emerald-200' };
    if (dice >= 0.85) return { label: 'Good Concordance', color: 'text-teal-700 bg-teal-50 border-teal-200' };
    return { label: 'Moderate Overlap', color: 'text-amber-700 bg-amber-50 border-amber-200' };
  };

  const diceQuality = getDiceQuality(metrics.dice);

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 p-4 sm:p-5 shadow-xs space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <div className="w-7 h-7 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold">
            <BarChart3 className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">
              Quantitative Evaluation & Inference Metrics
            </h3>
            <p className="text-xs text-slate-500">
              {metrics.metrics_source === 'test_set' ? 'Dice / IoU / Sens / Spec: checkpoint scores on the held-out test set. Confidence + latency: this image.' : 'Test-set scores not loaded (add bvi_net_<organ>_metrics.json next to the checkpoint).'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full border ${diceQuality.color}`}>
            <CheckCircle2 className="w-3 h-3" />
            <span>{diceQuality.label}</span>
          </span>
        </div>
      </div>

      {/* Primary 3 Key Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
        {/* 1. Dice Score */}
        <div className="bg-gradient-to-br from-teal-50/60 to-white rounded-xl border border-teal-100 p-4 shadow-2xs relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-500 mb-1.5">
            <span className="text-xs font-bold uppercase tracking-wider text-teal-800 flex items-center gap-1.5">
              <Target className="w-3.5 h-3.5 text-teal-600" />
              {metrics.metrics_source === 'test_set' ? 'Model Test-Set Dice (DSC)' : 'Dice Similarity Score (DSC)'}
            </span>
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl sm:text-3xl font-extrabold text-teal-950 font-mono tracking-tight">
              {(metrics.dice * 100).toFixed(2)}%
            </span>
            <span className="text-xs font-mono text-teal-700 font-semibold">
              ({metrics.dice.toFixed(4)})
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 flex items-center justify-between">
            <span>IoU / Jaccard: <strong className="text-slate-700 font-mono">{(metrics.iou * 100).toFixed(1)}%</strong></span>
            <span>Conf (this image): <strong className="text-slate-700 font-mono">{(metrics.confidence * 100).toFixed(1)}%</strong></span>
          </div>
        </div>

        {/* 2. Inference Time (ms) */}
        <div className="bg-gradient-to-br from-slate-50 to-white rounded-xl border border-slate-200 p-4 shadow-2xs">
          <div className="flex items-center justify-between text-slate-500 mb-1.5">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-500" />
              Inference Latency
            </span>
            <span className="text-[10px] font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200/60 font-mono">
              Live
            </span>
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl sm:text-3xl font-extrabold text-slate-900 font-mono tracking-tight">
              {metrics.inference_time_ms.toFixed(1)} <span className="text-sm font-sans font-semibold text-slate-500">ms</span>
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Compute: <strong className="text-slate-700 font-mono">{metrics.flops_gflops ? `${metrics.flops_gflops} GFLOPs` : 'n/a'}</strong></span>
            <span>FPS: <strong className="text-slate-700 font-mono">~{Math.round(1000 / metrics.inference_time_ms)} FPS</strong></span>
          </div>
        </div>

        {/* 3. Model Size / Parameter Count (Dynamic per organ) */}
        <div className="bg-gradient-to-br from-slate-50 to-white rounded-xl border border-slate-200 p-4 shadow-2xs">
          <div className="flex items-center justify-between text-slate-500 mb-1.5">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-teal-600" />
              Model Architecture Size
            </span>
            <span className="text-[10px] font-bold text-teal-700 bg-teal-50 px-1.5 py-0.5 rounded border border-teal-200/60 font-mono">
              Ultra-Compact
            </span>
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-xl sm:text-2xl font-extrabold text-slate-900 font-mono tracking-tight truncate">
              {metrics.model_size || currentOrgan.modelParams}
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Dataset: <strong className="text-slate-700 truncate max-w-[140px]">{currentOrgan.datasetName.split(':')[0]}</strong></span>
            <span className="text-teal-700 font-semibold font-mono">{(100 * (1 - metrics.parameter_count / 31.0e6)).toFixed(1)}% &darr; vs UNet</span>
          </div>
        </div>
      </div>

      {/* Secondary Efficiency Benchmark Footnote */}
      <div className="bg-slate-50 rounded-lg p-3 border border-slate-200/80 flex flex-col md:flex-row md:items-center justify-between gap-2 text-xs text-slate-600">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-teal-600 shrink-0" />
          <span>
            <strong>BVI-Net Efficiency:</strong> Employs Gabor directional filters and Mamba State Space sequence modeling to achieve <strong>~27k parameters</strong> compared to standard UNet (31.0M) and TransUNet (105.3M).
          </span>
        </div>
        <div className="flex items-center space-x-3 text-[11px] font-mono shrink-0">
          <span className="text-slate-500">Sensitivity: <strong className="text-slate-800">{(metrics.sensitivity * 100).toFixed(1)}%</strong></span>
          <span className="text-slate-500">Specificity: <strong className="text-slate-800">{(metrics.specificity * 100).toFixed(1)}%</strong></span>
        </div>
      </div>
    </div>
  );
};
