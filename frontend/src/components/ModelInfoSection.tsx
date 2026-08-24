import React from 'react';
import { OrganType } from '../types';
import { ORGAN_CONFIGS } from '../data/organData';
import {
  Cpu,
  Eye,
  GitBranch,
  Layers,
  Sparkles,
  Database,
  ChevronDown,
  ChevronUp,
  Activity,
  CheckCircle2,
  ExternalLink,
  Shield,
  Zap,
} from 'lucide-react';

interface ModelInfoSectionProps {
  selectedOrgan: OrganType;
  isOpen: boolean;
  onToggle: () => void;
}

export const ModelInfoSection: React.FC<ModelInfoSectionProps> = ({
  selectedOrgan,
  isOpen,
  onToggle,
}) => {
  const currentOrgan = ORGAN_CONFIGS[selectedOrgan];

  const components = [
    {
      id: 'gabor',
      title: 'Gabor Local Pathway',
      subtitle: 'Biological Early Vision & Margin Demarcation',
      badge: 'Bio-Visual V1 Cortex',
      icon: <Eye className="w-4 h-4 text-teal-600" />,
      description:
        'Inspired by the primary visual cortex (V1) in biological vision. Uses multi-scale, multi-directional directional Gabor filter banks to extract micro-textures, fine pigment networks, and subtle margin transitions with near-zero learnable parameter overhead.',
    },
    {
      id: 'mamba',
      title: 'Mamba-based Global Pathway',
      subtitle: 'Linear-Complexity State Space Model (SSM)',
      badge: 'O(N) Context',
      icon: <Zap className="w-4 h-4 text-amber-600" />,
      description:
        'Replaces quadratic Vision Transformer (ViT) self-attention with selective state space sequence modeling. Captures holistic global anatomical context across long spatial distances while maintaining linear O(N) computation and ultra-lightweight memory footprint.',
    },
    {
      id: 'gcn',
      title: 'GCN Attention (Graph Convolution)',
      subtitle: 'Topological Cross-Scale Feature Aggregation',
      badge: 'Relational Graph',
      icon: <GitBranch className="w-4 h-4 text-indigo-600" />,
      description:
        'Constructs an adaptive semantic graph over multi-scale feature tokens. Performs relational reasoning to resolve ambiguous lesion boundaries, specular dermoscopy glares, and low-contrast soft-tissue interfaces.',
    },
  ];

  const benchmarkData = [
    { model: 'BVI-Net (Ours)', params: '0.027 M (27k)', flops: '0.08 G', isicDice: '93.8%', litsDice: '90.8%', bratsDice: '91.5%', highlight: true },
    { model: 'Standard U-Net', params: '31.03 M', flops: '12.4 G', isicDice: '88.5%', litsDice: '85.2%', bratsDice: '86.4%', highlight: false },
    { model: 'TransUNet (ViT)', params: '105.28 M', flops: '24.8 G', isicDice: '91.7%', litsDice: '88.1%', bratsDice: '89.2%', highlight: false },
    { model: 'Swin-Unet', params: '27.17 M', flops: '6.1 G', isicDice: '91.2%', litsDice: '87.6%', bratsDice: '88.9%', highlight: false },
    { model: 'nnU-Net', params: '19.07 M', flops: '8.3 G', isicDice: '92.4%', litsDice: '89.4%', bratsDice: '90.1%', highlight: false },
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 shadow-xs overflow-hidden transition-all duration-200">
      {/* Collapsible Trigger Header */}
      <button
        id="model-info-accordion-toggle"
        onClick={onToggle}
        className="w-full px-5 py-4 flex items-center justify-between bg-white hover:bg-slate-50/80 transition-colors cursor-pointer text-left"
      >
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold">
            <Cpu className="w-4.5 h-4.5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-sm font-bold text-slate-900">
                Model Info & Bio-Visual Architecture
              </h3>
              <span className="text-[11px] bg-teal-50 text-teal-700 font-mono font-semibold px-2 py-0.5 rounded-full border border-teal-200/60">
                BVI-Net
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Gabor Local Pathway • Mamba Global Pathway • GCN Attention
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-xs font-semibold text-teal-700">
          <span>{isOpen ? 'Hide Details' : 'View Architecture'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {/* Expanded Content Panel */}
      {isOpen && (
        <div className="px-5 pb-6 pt-2 border-t border-slate-100 space-y-6">
          {/* Active Dataset Training Context */}
          <div className="bg-slate-50 rounded-xl p-4 border border-slate-200/80">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-teal-600" />
                Active Training Dataset & Cohort
              </span>
              <span className="text-xs font-mono font-semibold text-teal-800 bg-teal-100/70 px-2 py-0.5 rounded">
                {currentOrgan.shortLabel} Checkpoint
              </span>
            </div>
            <h4 className="text-sm font-bold text-slate-900 mb-1">
              {currentOrgan.datasetName}
            </h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              {currentOrgan.description}
            </p>
            <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <div className="bg-white p-2 rounded border border-slate-200">
                <span className="text-slate-400 block text-[10px]">Modality</span>
                <span className="font-semibold text-slate-800 truncate block">{currentOrgan.modalities.split('/')[0]}</span>
              </div>
              <div className="bg-white p-2 rounded border border-slate-200">
                <span className="text-slate-400 block text-[10px]">Model Parameters</span>
                <span className="font-semibold text-teal-700 font-mono block">{currentOrgan.modelParams.split(' ')[0]}</span>
              </div>
              <div className="bg-white p-2 rounded border border-slate-200">
                <span className="text-slate-400 block text-[10px]">Input Resolution</span>
                <span className="font-semibold text-slate-800 font-mono block">{currentOrgan.resolution}</span>
              </div>
              <div className="bg-white p-2 rounded border border-slate-200">
                <span className="text-slate-400 block text-[10px]">FLOPs</span>
                <span className="font-semibold text-slate-800 font-mono block">{currentOrgan.flops}</span>
              </div>
            </div>
          </div>

          {/* Key 3 Architectural Components */}
          <div>
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-3">
              Core Architectural Innovations
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
              {components.map((comp) => (
                <div
                  key={comp.id}
                  className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-2xs hover:border-teal-300 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="w-7 h-7 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-center">
                      {comp.icon}
                    </div>
                    <span className="text-[10px] font-mono font-bold bg-slate-100 text-slate-700 px-2 py-0.5 rounded-full">
                      {comp.badge}
                    </span>
                  </div>
                  <h5 className="text-xs font-bold text-slate-900">{comp.title}</h5>
                  <p className="text-[11px] font-medium text-teal-700 mb-1.5">{comp.subtitle}</p>
                  <p className="text-xs text-slate-600 leading-relaxed">{comp.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Comparative Benchmark Table */}
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                Benchmark Efficiency Comparison (Multi-Organ Segmentation)
              </h4>
              <span className="text-[11px] text-teal-700 font-semibold font-mono">
                99.9% Parameter Compression
              </span>
            </div>
            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3">Architecture</th>
                    <th className="py-2.5 px-3">Parameters (M)</th>
                    <th className="py-2.5 px-3">FLOPs</th>
                    <th className="py-2.5 px-3">Skin (ISIC) Dice</th>
                    <th className="py-2.5 px-3">Liver (LiTS) Dice</th>
                    <th className="py-2.5 px-3">Brain (BraTS) Dice</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {benchmarkData.map((row) => (
                    <tr
                      key={row.model}
                      className={
                        row.highlight
                          ? 'bg-teal-50/70 font-semibold text-teal-950'
                          : 'text-slate-700 hover:bg-slate-50/50'
                      }
                    >
                      <td className="py-2.5 px-3 flex items-center space-x-1.5">
                        {row.highlight && <Sparkles className="w-3.5 h-3.5 text-teal-600" />}
                        <span>{row.model}</span>
                      </td>
                      <td className="py-2.5 px-3 font-mono">{row.params}</td>
                      <td className="py-2.5 px-3 font-mono">{row.flops}</td>
                      <td className="py-2.5 px-3 font-mono text-teal-700 font-bold">{row.isicDice}</td>
                      <td className="py-2.5 px-3 font-mono">{row.litsDice}</td>
                      <td className="py-2.5 px-3 font-mono">{row.bratsDice}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
