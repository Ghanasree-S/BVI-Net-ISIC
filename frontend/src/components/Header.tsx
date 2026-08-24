import React from 'react';
import { Activity, Cpu, Layers, Sparkles, BookOpen, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  onOpenModelInfo: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onOpenModelInfo }) => {
  return (
    <header className="border-b border-slate-200/80 bg-white/90 backdrop-blur-md sticky top-0 z-30 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-18">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3.5">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-teal-600 to-cyan-700 flex items-center justify-center shadow-sm shadow-teal-700/20 text-white font-bold">
              <Activity className="w-6 h-6 stroke-[2.2]" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h1 className="text-lg sm:text-xl font-bold tracking-tight text-slate-900">
                  BVI-Net: <span className="text-teal-700 font-semibold">Multi-Organ Lesion Segmentation</span>
                </h1>
                <span className="hidden md:inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-teal-50 text-teal-700 border border-teal-200/70">
                  <ShieldCheck className="w-3 h-3" />
                  Clinical AI Demo
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-500 font-normal">
                Ultra-Lightweight Bio-Visually Inspired Network
              </p>
            </div>
          </div>

          {/* Right Status & Actions */}
          <div className="flex items-center space-x-3">
            <div className="hidden lg:flex items-center space-x-2 text-xs bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 text-slate-600">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="font-mono text-slate-700 font-medium">v1.0.4 Checkpoint Active</span>
              <span className="text-slate-300">|</span>
              <span className="text-slate-500">Gabor-Mamba-GCN</span>
            </div>

            <button
              id="model-info-btn"
              onClick={onOpenModelInfo}
              className="inline-flex items-center space-x-2 px-3.5 py-2 text-xs sm:text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200/80 active:bg-slate-200 border border-slate-200 rounded-lg transition-colors cursor-pointer"
              title="View BVI-Net Architecture and Parameter Details"
            >
              <BookOpen className="w-4 h-4 text-teal-600" />
              <span>Model Info</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
