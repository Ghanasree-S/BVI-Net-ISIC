import React from 'react';
import { PredictionResult } from '../types';
import { History, Clock, ArrowUpRight, Trash2, CheckCircle2, Sparkles } from 'lucide-react';

interface HistoryGalleryProps {
  history: PredictionResult[];
  activeResultId?: string;
  onSelectHistoryItem: (item: PredictionResult) => void;
  onClearHistory: () => void;
}

export const HistoryGallery: React.FC<HistoryGalleryProps> = ({
  history,
  activeResultId,
  onSelectHistoryItem,
  onClearHistory,
}) => {
  if (!history || history.length === 0) {
    return null;
  }

  const getOrganBadgeColor = (organ: string) => {
    switch (organ) {
      case 'skin':
        return 'bg-teal-50 text-teal-700 border-teal-200';
      case 'liver':
        return 'bg-amber-50 text-amber-800 border-amber-200';
      case 'brain':
        return 'bg-indigo-50 text-indigo-700 border-indigo-200';
      default:
        return 'bg-slate-100 text-slate-700 border-slate-200';
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 p-4 sm:p-5 shadow-xs">
      <div className="flex items-center justify-between mb-3.5 pb-2.5 border-b border-slate-100">
        <div className="flex items-center space-x-2">
          <div className="w-7 h-7 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold">
            <History className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-800">
              Recent Segmentation History
            </h3>
            <p className="text-xs text-slate-500">
              Click any previous session to review overlays and metrics
            </p>
          </div>
        </div>

        <button
          onClick={onClearHistory}
          className="inline-flex items-center space-x-1 text-xs text-slate-400 hover:text-red-600 transition-colors p-1 rounded cursor-pointer"
          title="Clear Gallery History"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Clear History</span>
        </button>
      </div>

      {/* Horizontal Strip of Thumbnails */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
        {history.slice(0, 5).map((item) => {
          const isActive = item.id === activeResultId;
          return (
            <button
              key={item.id}
              id={`history-item-${item.id}`}
              onClick={() => onSelectHistoryItem(item)}
              className={`group relative text-left rounded-xl p-2.5 border transition-all cursor-pointer flex flex-col ${
                isActive
                  ? 'bg-teal-50/70 border-teal-500 ring-2 ring-teal-500/20 shadow-xs'
                  : 'bg-slate-50/70 border-slate-200 hover:bg-white hover:border-slate-300 hover:shadow-xs'
              }`}
            >
              {/* Dual Thumbnail Preview (Original + Mask) */}
              <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-slate-950 border border-slate-200/80 mb-2">
                <img
                  src={item.original_image_url}
                  alt="Original Thumbnail"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                />
                <div className="absolute bottom-1 right-1 w-7 h-7 rounded border border-white/60 overflow-hidden bg-black shadow-xs">
                  <img
                    src={item.mask_base64}
                    alt="Mask Thumbnail"
                    className="w-full h-full object-cover"
                  />
                </div>

                {isActive && (
                  <div className="absolute top-1 left-1 bg-teal-600 text-white rounded-full p-0.5">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  </div>
                )}
              </div>

              {/* Tag and Metrics */}
              <div className="flex-1 flex flex-col justify-between space-y-1">
                <div className="flex items-center justify-between">
                  <span
                    className={`text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded border capitalize ${getOrganBadgeColor(
                      item.organ_type
                    )}`}
                  >
                    {item.organ_type}
                  </span>
                  <span className="text-[11px] font-mono font-bold text-teal-800">
                    {(item.metrics.dice * 100).toFixed(1)}% DSC
                  </span>
                </div>

                <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                  <span>{item.metrics.inference_time_ms.toFixed(1)}ms</span>
                  <span>{item.timestamp.split(' ')[0]}</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
