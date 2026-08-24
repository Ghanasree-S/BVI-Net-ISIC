import React, { useEffect, useState } from 'react';
import { OrganType } from '../types';
import { ORGAN_CONFIGS } from '../data/organData';
import { Activity, Cpu, Layers, Sparkles, Zap, Eye, GitBranch } from 'lucide-react';

interface LoadingSkeletonProps {
  organType: OrganType;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ organType }) => {
  const currentOrgan = ORGAN_CONFIGS[organType];
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { label: 'Ingesting 256×256 matrix & Bio-Visual Normalization', icon: <Eye className="w-3.5 h-3.5" /> },
    { label: 'Gabor Filter Bank Directional Feature Extraction', icon: <Cpu className="w-3.5 h-3.5" /> },
    { label: 'Mamba Selective State Space Global Pathway', icon: <Zap className="w-3.5 h-3.5" /> },
    { label: 'GCN Attention Lesion Boundary Aggregation', icon: <GitBranch className="w-3.5 h-3.5" /> },
  ];

  useEffect(() => {
    const timer1 = setTimeout(() => setCurrentStep(1), 100);
    const timer2 = setTimeout(() => setCurrentStep(2), 220);
    const timer3 = setTimeout(() => setCurrentStep(3), 360);
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, []);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-6">
      {/* Header with spinner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-teal-50 border border-teal-200 text-teal-700 flex items-center justify-center relative">
            <Activity className="w-5 h-5 animate-spin" />
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-slate-900">
              Running BVI-Net Neural Inference...
            </h3>
            <p className="text-xs text-slate-500">
              Target: <span className="font-semibold text-teal-800">{currentOrgan.name}</span> • Lightweight Checkpoint
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-xs font-mono bg-teal-50/70 text-teal-800 px-3 py-1.5 rounded-lg border border-teal-200">
          <span className="w-2 h-2 rounded-full bg-teal-500 animate-ping"></span>
          <span>Latency &lt; 20ms Target</span>
        </div>
      </div>

      {/* Step progress pills */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
        {steps.map((step, idx) => {
          const isDone = idx < currentStep;
          const isCurrent = idx === currentStep;
          return (
            <div
              key={idx}
              className={`p-2.5 rounded-lg text-xs font-medium border flex items-center space-x-2 transition-all ${
                isDone
                  ? 'bg-teal-50 text-teal-900 border-teal-200'
                  : isCurrent
                  ? 'bg-white text-teal-700 border-teal-400 ring-2 ring-teal-500/10'
                  : 'bg-slate-50 text-slate-400 border-slate-200 opacity-60'
              }`}
            >
              <div
                className={`w-5 h-5 rounded flex items-center justify-center shrink-0 ${
                  isDone || isCurrent ? 'text-teal-700' : 'text-slate-400'
                }`}
              >
                {step.icon}
              </div>
              <span className="truncate text-[11px]">{step.label}</span>
            </div>
          );
        })}
      </div>

      {/* Triple image pulse skeletons */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[1, 2, 3].map((n) => (
          <div key={n} className="rounded-xl border border-slate-200 bg-slate-50 p-3 space-y-3">
            <div className="h-4 w-28 bg-slate-200 rounded animate-pulse"></div>
            <div className="h-56 bg-slate-200 rounded-lg animate-pulse flex items-center justify-center">
              <Layers className="w-8 h-8 text-slate-300 animate-bounce" />
            </div>
            <div className="h-3 w-36 bg-slate-200 rounded animate-pulse"></div>
          </div>
        ))}
      </div>
    </div>
  );
};
