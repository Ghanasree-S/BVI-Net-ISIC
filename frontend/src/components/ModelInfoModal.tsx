import React from 'react';
import { OrganType } from '../types';
import { ORGAN_CONFIGS } from '../data/organData';
import {
  X,
  Cpu,
  Eye,
  GitBranch,
  Layers,
  Sparkles,
  Database,
  ShieldCheck,
  Zap,
  Activity,
} from 'lucide-react';

interface ModelInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedOrgan: OrganType;
}

export const ModelInfoModal: React.FC<ModelInfoModalProps> = ({
  isOpen,
  onClose,
  selectedOrgan,
}) => {
  if (!isOpen) return null;

  const currentOrgan = ORGAN_CONFIGS[selectedOrgan];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs animate-in fade-in duration-150">
      <div
        className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-slate-200 flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="p-5 border-b border-slate-200 flex items-center justify-between sticky top-0 bg-white z-10">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-teal-600 text-white flex items-center justify-center font-bold shadow-xs">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                BVI-Net Architecture Specification
              </h3>
              <p className="text-xs text-slate-500">
                Bio-Visually Inspired Multi-Organ Lesion Segmentation Network
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 text-slate-700 text-xs sm:text-sm">
          {/* Executive Summary */}
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
            <h4 className="text-xs font-bold text-teal-800 uppercase tracking-wider mb-1 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-teal-600" />
              Why BVI-Net?
            </h4>
            <p className="text-slate-600 leading-relaxed text-xs">
              Conventional medical vision architectures (like 31M parameter U-Net or 105M parameter TransUNet) suffer from extreme computational weight and high memory consumption. <strong>BVI-Net</strong> introduces a biological early-vision paradigm combining <em>multi-directional Gabor filters</em>, <em>linear state-space sequence modeling (Mamba)</em>, and <em>graph relational attention (GCN)</em> to achieve state-of-the-art segmentation with only <strong>27,312 parameters (0.11 MB)</strong>.
            </p>
          </div>

          {/* Tri-Pathway Architecture */}
          <div>
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3">
              Tri-Pathway Architectural Breakdown
            </h4>

            <div className="space-y-3">
              {/* 1. Gabor Local Pathway */}
              <div className="p-3.5 rounded-xl border border-slate-200 bg-white shadow-2xs">
                <div className="flex items-center space-x-2 text-slate-900 font-bold mb-1">
                  <Eye className="w-4 h-4 text-teal-600" />
                  <span>1. Gabor Local Pathway (V1 Cortex Receptive Fields)</span>
                </div>
                <p className="text-slate-600 text-xs leading-relaxed">
                  Employs mathematically parameterized Gabor wavelet kernels oriented across &theta; &isin; &#123;0&deg;, 45&deg;, 90&deg;, 135&deg;&#125; with multiscale frequencies. Isolates high-frequency boundary gradients, dermoscopic pigment networks, and lesion margin transitions without heavy learned weights.
                </p>
              </div>

              {/* 2. Mamba-based Global Pathway */}
              <div className="p-3.5 rounded-xl border border-slate-200 bg-white shadow-2xs">
                <div className="flex items-center space-x-2 text-slate-900 font-bold mb-1">
                  <Zap className="w-4 h-4 text-amber-600" />
                  <span>2. Mamba-based Global Pathway (Linear State-Space Sequence Modeling)</span>
                </div>
                <p className="text-slate-600 text-xs leading-relaxed">
                  Leverages continuous-time State Space Models (SSM) discretized through selective parameter scanning. Enables global anatomical context propagation across the entire scan with O(N) complexity instead of standard Transformer O(N&sup2;) self-attention.
                </p>
              </div>

              {/* 3. GCN Attention */}
              <div className="p-3.5 rounded-xl border border-slate-200 bg-white shadow-2xs">
                <div className="flex items-center space-x-2 text-slate-900 font-bold mb-1">
                  <GitBranch className="w-4 h-4 text-indigo-600" />
                  <span>3. GCN Cross-Scale Relational Attention</span>
                </div>
                <p className="text-slate-600 text-xs leading-relaxed">
                  Projects multi-level feature maps into a topological graph where nodes represent semantic tissue regions and edges represent spatial adjacency. Aggregates multi-scale contextual features to eliminate spurious false-positive detections.
                </p>
              </div>
            </div>
          </div>

          {/* Dataset Status */}
          <div className="bg-teal-50/70 border border-teal-200/80 rounded-xl p-4">
            <h4 className="text-xs font-bold text-teal-900 uppercase tracking-wider mb-1 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-teal-700" />
              Multi-Organ Datasets & Checkpoint Readiness
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 mt-3 text-xs">
              <div className="bg-white p-2.5 rounded-lg border border-teal-100 shadow-2xs">
                <div className="font-bold text-teal-900">Skin (ISIC2018)</div>
                <span className="text-[10px] bg-emerald-100 text-emerald-800 font-semibold px-1.5 py-0.2 rounded inline-block my-1">
                  Active Ready
                </span>
                <p className="text-[11px] text-slate-500">2,594 Dermoscopy images • 27k params</p>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-slate-200 shadow-2xs">
                <div className="font-bold text-slate-800">Liver (LiTS17)</div>
                <span className="text-[10px] bg-amber-100 text-amber-800 font-semibold px-1.5 py-0.2 rounded inline-block my-1">
                  Coming Soon
                </span>
                <p className="text-[11px] text-slate-500">Abdominal CT 3D Slices • 31k params</p>
              </div>
              <div className="bg-white p-2.5 rounded-lg border border-slate-200 shadow-2xs">
                <div className="font-bold text-slate-800">Brain (BraTS19)</div>
                <span className="text-[10px] bg-amber-100 text-amber-800 font-semibold px-1.5 py-0.2 rounded inline-block my-1">
                  Coming Soon
                </span>
                <p className="text-[11px] text-slate-500">Multimodal MRI Volumes • 34k params</p>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between">
          <div className="flex items-center space-x-1.5 text-xs text-slate-500">
            <ShieldCheck className="w-4 h-4 text-teal-600" />
            <span>Research & Clinical Diagnostic Evaluation License</span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-900 text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
          >
            Close Window
          </button>
        </div>
      </div>
    </div>
  );
};
