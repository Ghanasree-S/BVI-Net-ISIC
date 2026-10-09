import React, { useState, useEffect } from 'react';
import { OrganType, PredictionResult, SampleCase } from './types';
import { ORGAN_CONFIGS } from './data/organData';
import { predictLesion } from './services/api';
import { Header } from './components/Header';
import { OrganSelector } from './components/OrganSelector';
import { ImageUploadArea } from './components/ImageUploadArea';
import { ResultsViewer } from './components/ResultsViewer';
import { StatsPanel } from './components/StatsPanel';
import { ModelInfoSection } from './components/ModelInfoSection';
import { HistoryGallery } from './components/HistoryGallery';
import { LoadingSkeleton } from './components/LoadingSkeleton';
import { ModelInfoModal } from './components/ModelInfoModal';
import {
  Activity,
  Sparkles,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
  Info,
  Sliders,
  CheckCircle2,
} from 'lucide-react';

export default function App() {
  const [selectedOrgan, setSelectedOrgan] = useState<OrganType>('skin');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activeResult, setActiveResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<PredictionResult[]>([]);
  const [isModelInfoOpen, setIsModelInfoOpen] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Pre-load default sample on initial load if desired, or show clean empty state
  const handleOrganChange = (organ: OrganType) => {
    setSelectedOrgan(organ);
  };

  // Perform segmentation prediction
  const handleProcessImage = async (imageBase64: string, fileName: string) => {
    setIsLoading(true);
    setPredictError(null);
    try {
      const result = await predictLesion(selectedOrgan, imageBase64, fileName);
      setActiveResult(result);
      
      // Add to history strip (keep latest 5)
      setHistory((prev) => {
        const filtered = prev.filter((item) => item.id !== result.id);
        return [result, ...filtered].slice(0, 5);
      });
    } catch (error) {
      console.error('Prediction failed:', error);
      setPredictError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadSample = (sample: SampleCase) => {
    handleProcessImage(sample.imageUrl, `${sample.title}${sample.imageUrl.endsWith('.npy') ? '.npy' : '.png'}`);
  };

  const handleSelectHistoryItem = (item: PredictionResult) => {
    setSelectedOrgan(item.organ_type);
    setActiveResult(item);
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  const handleReset = () => {
    setActiveResult(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col selection:bg-teal-500/20 selection:text-teal-950 text-slate-900 font-sans">
      {/* Top Header */}
      <Header onOpenModelInfo={() => setIsModalOpen(true)} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6">
        {/* Organ Selector Bar */}
        <OrganSelector
          selectedOrgan={selectedOrgan}
          onSelectOrgan={handleOrganChange}
        />

        {predictError && !isLoading && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 flex items-start justify-between gap-3">
            <span><strong>Prediction failed:</strong> {predictError}</span>
            <button className="text-red-600 hover:text-red-800 font-semibold shrink-0" onClick={() => setPredictError(null)}>Dismiss</button>
          </div>
        )}

        {/* Loading State */}
        {isLoading && <LoadingSkeleton organType={selectedOrgan} />}

        {/* When a result is active, show the 3-view display and stats */}
        {!isLoading && activeResult && (
          <div className="space-y-6 animate-in fade-in duration-200">
            {/* Triple Side-by-Side View */}
            <ResultsViewer
              result={activeResult}
              isLoading={isLoading}
              onReset={handleReset}
            />

            {/* Quantitative Stats Panel */}
            <StatsPanel
              metrics={activeResult.metrics}
              organType={activeResult.organ_type}
            />
          </div>
        )}

        {/* Upload Zone (Visible when no active result or can be toggled) */}
        {!isLoading && !activeResult && (
          <div className="space-y-6 animate-in fade-in duration-200">
            {/* Image Upload Dropzone & Sample presets */}
            <ImageUploadArea
              selectedOrgan={selectedOrgan}
              isLoading={isLoading}
              onImageSelected={handleProcessImage}
              onLoadSample={handleLoadSample}
            />
          </div>
        )}

        {/* Collapsible Model Info & Architecture Breakdown */}
        <ModelInfoSection
          selectedOrgan={selectedOrgan}
          isOpen={isModelInfoOpen}
          onToggle={() => setIsModelInfoOpen(!isModelInfoOpen)}
        />

        {/* Segmentation History Gallery Strip (bottom) */}
        <HistoryGallery
          history={history}
          activeResultId={activeResult?.id}
          onSelectHistoryItem={handleSelectHistoryItem}
          onClearHistory={handleClearHistory}
        />
      </main>

      {/* Hospital AI Dashboard Clinical Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-4 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-center sm:text-left">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-teal-500"></span>
            <span className="font-semibold text-slate-700">
              BVI-Net Research Demo
            </span>
            <span className="text-slate-300">•</span>
            <span>Ultra-Lightweight Bio-Visually Inspired Multi-Organ Segmentation</span>
          </div>

          <div className="flex items-center space-x-4 font-mono text-[11px] text-slate-400">
            <span>ISIC2018 (Skin)</span>
            <span>LiTS17 (Liver)</span>
            <span>BraTS19 (Brain)</span>
            <span>27k Params</span>
          </div>
        </div>
      </footer>

      {/* Model Info Modal Dialog */}
      <ModelInfoModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        selectedOrgan={selectedOrgan}
      />
    </div>
  );
}
