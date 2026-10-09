import React, { useRef, useState } from 'react';
import { OrganType, SampleCase } from '../types';
import { ORGAN_CONFIGS } from '../data/organData';
import {
  UploadCloud,
  FileImage,
  Sparkles,
  AlertCircle,
  Clock,
  ArrowRight,
  ShieldAlert,
  CheckCircle,
  HelpCircle,
} from 'lucide-react';

interface ImageUploadAreaProps {
  selectedOrgan: OrganType;
  isLoading: boolean;
  onImageSelected: (imageBase64: string, fileName: string) => void;
  onLoadSample: (sample: SampleCase) => void;
}

export const ImageUploadArea: React.FC<ImageUploadAreaProps> = ({
  selectedOrgan,
  isLoading,
  onImageSelected,
  onLoadSample,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const currentConfig = ORGAN_CONFIGS[selectedOrgan];

  const handleFile = (file: File) => {
    setErrorMsg(null);
    if (!file) return;

    // Verify format
    const isNpy = /\.npy$/i.test(file.name);
    if (selectedOrgan === 'brain' && !isNpy) {
      // The brain model needs 4 stacked MRI modalities, which a single image can't hold.
      setErrorMsg('Brain model needs a 4-channel .npy slice (T1, T1ce, T2, FLAIR) from data/prepare_brats.py.');
      return;
    }
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!isNpy && !validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i)) {
      setErrorMsg('Please upload a valid medical image file (JPG, PNG, or WEBP).');
      return;
    }

    // Size limit check (25MB)
    if (file.size > 25 * 1024 * 1024) {
      setErrorMsg('Image size exceeds 25MB limit. Please provide a standard scan slice.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      if (result) {
        onImageSelected(result, file.name);
      }
    };
    reader.onerror = () => {
      setErrorMsg('Failed to read image file. Please try again.');
    };
    reader.readAsDataURL(file);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (isLoading) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isLoading) setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const handleBrowseClick = () => {
    if (!isLoading && fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  return (
    <div className="space-y-4">
      {/* "Coming Soon" Graceful Notice for Liver & Brain */}
      {!currentConfig.isAvailable && (
        <div className="bg-amber-50/80 border border-amber-200/80 rounded-xl p-3.5 sm:p-4 text-amber-900 shadow-2xs">
          <div className="flex items-start space-x-3">
            <div className="w-6 h-6 rounded-md bg-amber-100 text-amber-700 flex items-center justify-center shrink-0 mt-0.5">
              <Clock className="w-4 h-4" />
            </div>
            <div className="flex-1 text-xs sm:text-sm">
              <div className="font-semibold text-amber-950 flex items-center gap-2">
                <span>{currentConfig.name} Checkpoint Notice</span>
                <span className="text-[11px] bg-amber-200 text-amber-900 px-2 py-0.2 rounded-full font-bold">
                  In Progress
                </span>
              </div>
              <p className="mt-0.5 text-amber-800 text-xs">
                The full 3D volumetric model for {currentConfig.shortLabel} is currently in final validation.
                You can upload a slice to test the <strong>Experimental Checkpoint Preview</strong> below, or select the active <strong>Skin (ISIC2018)</strong> model for production inference.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Upload Drop Zone Card */}
      <div
        id="medical-image-dropzone"
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={handleBrowseClick}
        className={`relative border-2 border-dashed rounded-2xl p-6 sm:p-8 text-center transition-all duration-200 cursor-pointer overflow-hidden ${
          isLoading
            ? 'bg-slate-50 border-slate-300 opacity-80 cursor-wait'
            : isDragging
            ? 'bg-teal-50/80 border-teal-500 ring-4 ring-teal-500/10 scale-[1.005]'
            : 'bg-white hover:bg-slate-50/60 border-slate-300 hover:border-teal-500/70 shadow-xs'
        }`}
      >
        <input
          ref={fileInputRef}
          id="medical-file-input"
          type="file"
          accept={selectedOrgan === 'brain' ? '.npy' : 'image/jpeg,image/png,image/webp,image/bmp'}
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        <div className="max-w-md mx-auto flex flex-col items-center">
          {/* Animated upload icon */}
          <div
            className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-3 transition-transform duration-200 ${
              isDragging
                ? 'bg-teal-600 text-white scale-110 shadow-md'
                : 'bg-teal-50 text-teal-700 border border-teal-100 group-hover:scale-105'
            }`}
          >
            <UploadCloud className="w-7 h-7" />
          </div>

          <h3 className="text-base font-bold text-slate-800 tracking-tight">
            {isDragging ? 'Drop your scan to analyze' : 'Upload Medical Image for Segmentation'}
          </h3>

          <p className="text-xs sm:text-sm text-slate-500 mt-1 mb-3">
            <span className="font-medium text-teal-700">{currentConfig.inputHint}</span>
          </p>

          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 text-xs font-semibold hover:bg-slate-200 transition-colors">
            <FileImage className="w-3.5 h-3.5 text-slate-500" />
            <span>Browse Computer (JPG, PNG)</span>
          </div>

          <div className="mt-3 flex items-center space-x-4 text-[11px] text-slate-400 font-mono">
            <span>• Max file size: 25MB</span>
            <span>• Auto-resized to {currentConfig.resolution}</span>
            <span>• HIPAA Client-Side Safe</span>
          </div>
        </div>

        {/* Error Banner */}
        {errorMsg && (
          <div className="mt-4 p-2.5 bg-red-50 border border-red-200 rounded-lg text-red-700 text-xs flex items-center justify-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {/* Preset Sample Clinical Scans for Instant 1-Click Verification */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-4 sm:p-4.5 shadow-xs">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-teal-600" />
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Instant Sample Scans ({currentConfig.shortLabel} Benchmark)
            </span>
          </div>
          <span className="text-[11px] text-slate-500 font-medium">
            1-Click Test Cases
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {currentConfig.sampleCases.map((sample) => (
            <button
              key={sample.id}
              id={`sample-btn-${sample.id}`}
              onClick={(e) => {
                e.stopPropagation();
                if (!isLoading) onLoadSample(sample);
              }}
              disabled={isLoading}
              className="group text-left flex items-center space-x-3 p-2.5 rounded-lg border border-slate-200/80 bg-slate-50/70 hover:bg-teal-50/50 hover:border-teal-300 transition-all cursor-pointer disabled:opacity-60"
            >
              <div className="w-13 h-13 rounded-lg overflow-hidden border border-slate-300 shrink-0 bg-slate-900 relative shadow-2xs">
                <img
                  src={sample.thumbnailUrl || sample.imageUrl}
                  alt={sample.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-slate-900 truncate group-hover:text-teal-800">
                    {sample.title}
                  </h4>
                  <span className="text-[10px] text-teal-700 font-mono font-medium shrink-0 ml-1">
                    {sample.resolution}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 line-clamp-1 mt-0.5">
                  {sample.description}
                </p>
                <div className="flex items-center space-x-1 text-[10px] text-slate-400 mt-1 font-mono">
                  <span>{sample.subType}</span>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-300 group-hover:text-teal-600 group-hover:translate-x-0.5 transition-all shrink-0" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
