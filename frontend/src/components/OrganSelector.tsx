import React from 'react';
import { OrganType } from '../types';
import { ORGAN_CONFIGS } from '../data/organData';
import { Sparkles, Info, CheckCircle2, Clock, Eye, Layers } from 'lucide-react';

interface OrganSelectorProps {
  selectedOrgan: OrganType;
  onSelectOrgan: (organ: OrganType) => void;
}

export const OrganSelector: React.FC<OrganSelectorProps> = ({
  selectedOrgan,
  onSelectOrgan,
}) => {
  const currentConfig = ORGAN_CONFIGS[selectedOrgan];

  const organList: Array<{ id: OrganType; label: string; badge: string; isFunctional: boolean }> = [
    {
      id: 'skin',
      label: 'Skin (ISIC2018)',
      badge: 'Active Ready',
      isFunctional: true,
    },
    {
      id: 'liver',
      label: 'Liver (LiTS17)',
      badge: 'Coming Soon',
      isFunctional: false,
    },
    {
      id: 'brain',
      label: 'Brain (BraTS19)',
      badge: 'Coming Soon',
      isFunctional: false,
    },
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 p-4 sm:p-5 shadow-xs">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
        <div>
          <span className="text-xs font-semibold text-teal-700 uppercase tracking-wider block mb-1">
            Target Organ & Benchmark Dataset
          </span>
          <h2 className="text-base sm:text-lg font-bold text-slate-800">
            Select Checkpoint Architecture
          </h2>
        </div>

        {/* Organ Selector Segmented Tabs */}
        <div
          role="tablist"
          aria-label="Target Organ Selection"
          className="inline-flex p-1 bg-slate-100/90 rounded-xl border border-slate-200 self-start md:self-auto overflow-x-auto max-w-full"
        >
          {organList.map((item) => {
            const isSelected = selectedOrgan === item.id;
            return (
              <button
                key={item.id}
                id={`organ-tab-${item.id}`}
                role="tab"
                aria-selected={isSelected}
                onClick={() => onSelectOrgan(item.id)}
                className={`group relative flex items-center space-x-2 px-3.5 sm:px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all duration-150 whitespace-nowrap cursor-pointer ${
                  isSelected
                    ? 'bg-white text-teal-900 shadow-xs ring-1 ring-slate-900/5'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white/50'
                }`}
              >
                <span>{item.label}</span>
                {item.isFunctional ? (
                  <span
                    className={`inline-flex items-center text-[10px] px-1.5 py-0.5 rounded-full font-bold uppercase tracking-wider ${
                      isSelected
                        ? 'bg-teal-100 text-teal-800'
                        : 'bg-emerald-50 text-emerald-700 border border-emerald-200/60'
                    }`}
                  >
                    Active
                  </span>
                ) : (
                  <span
                    className={`inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded-full font-bold tracking-wider ${
                      isSelected
                        ? 'bg-amber-100 text-amber-900'
                        : 'bg-slate-200 text-slate-600'
                    }`}
                  >
                    <Clock className="w-2.5 h-2.5" />
                    Soon
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Dynamic Hint & Dataset Details Bar */}
      <div className="mt-3.5 pt-1 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
        <div className="flex items-start sm:items-center space-x-2 text-slate-600">
          <div className="w-5 h-5 rounded-md bg-teal-50 text-teal-700 flex items-center justify-center shrink-0 mt-0.5 sm:mt-0">
            <Info className="w-3.5 h-3.5" />
          </div>
          <div>
            <span className="font-semibold text-slate-800">Upload Requirement: </span>
            <span className="text-slate-600 font-medium">{currentConfig.inputHint}</span>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-slate-500 font-mono text-[11px] shrink-0">
          <span className="inline-flex items-center gap-1 bg-slate-50 px-2 py-1 rounded border border-slate-200">
            <Layers className="w-3 h-3 text-teal-600" />
            <span className="text-slate-700 font-semibold">{currentConfig.modelParams}</span>
          </span>
          <span className="hidden lg:inline bg-slate-50 px-2 py-1 rounded border border-slate-200 text-slate-600">
            Res: {currentConfig.resolution}
          </span>
        </div>
      </div>
    </div>
  );
};
