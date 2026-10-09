import React, { useState, useEffect, useRef } from 'react';
import { PredictionResult, OverlayColor } from '../types';
import {
  Download,
  Sliders,
  Layers,
  ZoomIn,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  RefreshCw,
  Sparkles,
  Contrast,
  Check,
  SplitSquareVertical,
} from 'lucide-react';

interface ResultsViewerProps {
  result: PredictionResult;
  isLoading: boolean;
  onReset: () => void;
}

export const ResultsViewer: React.FC<ResultsViewerProps> = ({
  result,
  isLoading,
  onReset,
}) => {
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.55);
  const [overlayColor, setOverlayColor] = useState<OverlayColor>('red');
  const [showContourOnly, setShowContourOnly] = useState<boolean>(false);
  const [splitPosition, setSplitPosition] = useState<number>(50); // percentage for split view
  const [isSplitMode, setIsSplitMode] = useState<boolean>(false);
  const [isInvertedMask, setIsInvertedMask] = useState<boolean>(false);
  const [isCopied, setIsCopied] = useState<boolean>(false);

  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [renderedOverlayDataUrl, setRenderedOverlayDataUrl] = useState<string>('');

  // Generate composite high-resolution canvas overlay
  useEffect(() => {
    if (!result?.original_image_url || !result?.mask_base64) return;

    const baseImg = new Image();
    const maskImg = new Image();
    baseImg.crossOrigin = 'anonymous';
    maskImg.crossOrigin = 'anonymous';

    let loadedCount = 0;
    const onBothLoaded = () => {
      loadedCount++;
      if (loadedCount < 2) return;

      const width = baseImg.naturalWidth || 256;
      const height = baseImg.naturalHeight || 256;

      const canvas = overlayCanvasRef.current || document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // 1. Draw base original image
      ctx.drawImage(baseImg, 0, 0, width, height);

      // 2. Extract mask pixel data to apply color tint
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) return;

      tempCtx.drawImage(maskImg, 0, 0, width, height);
      const maskData = tempCtx.getImageData(0, 0, width, height);
      const pixels = maskData.data;

      // Tint color RGB values
      const colorMap: Record<OverlayColor, [number, number, number]> = {
        red: [239, 68, 68], // #ef4444 Crimson
        teal: [13, 148, 136], // #0d9488 Medical Teal
        emerald: [16, 185, 129], // #10b981 Emerald
        amber: [245, 158, 11], // #f59e0b Amber
      };
      const [r, g, b] = colorMap[overlayColor];

      const overlayImgData = ctx.createImageData(width, height);
      const outPixels = overlayImgData.data;

      for (let i = 0; i < pixels.length; i += 4) {
        // Mask is black/white. White (r > 120) is lesion area
        const maskVal = pixels[i];
        if (maskVal > 100) {
          outPixels[i] = r;
          outPixels[i + 1] = g;
          outPixels[i + 2] = b;
          outPixels[i + 3] = Math.round(overlayOpacity * 255);
        } else {
          outPixels[i + 3] = 0;
        }
      }

      // Draw overlay onto temporary canvas
      tempCtx.putImageData(overlayImgData, 0, 0);

      // Composite onto main canvas
      ctx.drawImage(tempCanvas, 0, 0);

      // Add crisp perimeter contour if enabled
      if (showContourOnly) {
        ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = 'rgba(0,0,0,0.5)';
        ctx.shadowBlur = 3;
      }

      setRenderedOverlayDataUrl(canvas.toDataURL('image/png'));
    };

    baseImg.onload = onBothLoaded;
    maskImg.onload = onBothLoaded;
    baseImg.src = result.original_image_url;
    maskImg.src = result.mask_base64;
  }, [result, overlayOpacity, overlayColor, showContourOnly]);

  const downloadImage = (dataUrl: string, fileName: string) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const colorOptions: Array<{ id: OverlayColor; label: string; bgClass: string }> = [
    { id: 'red', label: 'Crimson Red', bgClass: 'bg-red-500 ring-red-300' },
    { id: 'teal', label: 'Medical Teal', bgClass: 'bg-teal-500 ring-teal-300' },
    { id: 'emerald', label: 'Emerald Green', bgClass: 'bg-emerald-500 ring-emerald-300' },
    { id: 'amber', label: 'Safety Amber', bgClass: 'bg-amber-500 ring-amber-300' },
  ];

  return (
    <div className="space-y-4">
      {/* Control Header & Overlay Customization Bar */}
      <div className="bg-white rounded-xl border border-slate-200/90 p-4 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold">
              <Layers className="w-4.5 h-4.5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-800">
                Triple-View Segmentation Output
              </h3>
              <p className="text-xs text-slate-500">
                Original Scan • Predicted Lesion Mask • Calibrated Overlay
              </p>
            </div>
          </div>

          {/* Interactive Overlay Customization Controls */}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            {/* Color Tint Selector */}
            <div className="flex items-center space-x-1.5 bg-slate-50 p-1.5 rounded-lg border border-slate-200">
              <span className="text-slate-500 font-medium px-1">Tint:</span>
              {colorOptions.map((opt) => (
                <button
                  key={opt.id}
                  id={`overlay-color-${opt.id}`}
                  onClick={() => setOverlayColor(opt.id)}
                  title={opt.label}
                  className={`w-5 h-5 rounded-full transition-all cursor-pointer ${opt.bgClass} ${
                    overlayColor === opt.id
                      ? 'ring-2 ring-offset-1 scale-110 shadow-xs'
                      : 'opacity-70 hover:opacity-100'
                  }`}
                />
              ))}
            </div>

            {/* Opacity Slider */}
            <div className="flex items-center space-x-2 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">
              <Sliders className="w-3.5 h-3.5 text-slate-500" />
              <span className="text-slate-600 font-medium">Opacity:</span>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-20 h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-teal-600"
              />
              <span className="font-mono text-slate-700 font-semibold w-8 text-right">
                {Math.round(overlayOpacity * 100)}%
              </span>
            </div>

            {/* Split Comparison Toggle */}
            <button
              id="split-view-toggle-btn"
              onClick={() => setIsSplitMode(!isSplitMode)}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium border transition-colors cursor-pointer ${
                isSplitMode
                  ? 'bg-teal-50 text-teal-800 border-teal-300'
                  : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
              }`}
            >
              <SplitSquareVertical className="w-3.5 h-3.5" />
              <span>{isSplitMode ? 'Exit Split View' : 'Interactive Split'}</span>
            </button>

            {/* Reset / New Scan */}
            <button
              id="new-scan-btn"
              onClick={onReset}
              className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 font-medium border border-slate-200 transition-colors cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>New Scan</span>
            </button>
          </div>
        </div>
      </div>

      {/* Triple Side-by-Side Images Display */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* 1. ORIGINAL IMAGE */}
        <div className="bg-white rounded-xl border border-slate-200/90 shadow-xs overflow-hidden flex flex-col">
          <div className="bg-slate-50/90 px-3.5 py-2.5 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-slate-400"></span>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                1. Original Image
              </h4>
            </div>
            <span className="text-[11px] font-mono text-slate-500 bg-white px-2 py-0.5 rounded border border-slate-200">
              Input: {result.model_info?.input_resolution || 'RGB'}
            </span>
          </div>

          <div className="p-3 bg-slate-950 flex-1 flex items-center justify-center min-h-[260px] sm:min-h-[300px] relative group">
            <img
              src={result.original_image_url}
              alt="Original Medical Image"
              className="max-h-[320px] w-full object-contain rounded-lg transition-transform duration-200 group-hover:scale-[1.01]"
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/60 backdrop-blur-sm rounded text-[10px] text-slate-300 font-mono">
              Raw Pixel Matrix
            </div>
          </div>

          <div className="p-2.5 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500">
            <span className="truncate max-w-[180px]">{result.fileName || 'Original Image'}</span>
            <button
              onClick={() => downloadImage(result.original_image_url, `original_${result.fileName || 'scan.png'}`)}
              className="p-1 hover:text-teal-700 rounded hover:bg-white transition-colors"
              title="Download Original Image"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* 2. PREDICTED MASK (Black & White) */}
        <div className="bg-white rounded-xl border border-slate-200/90 shadow-xs overflow-hidden flex flex-col">
          <div className="bg-slate-50/90 px-3.5 py-2.5 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-teal-500"></span>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                2. Predicted Mask
              </h4>
            </div>
            <div className="flex items-center space-x-1.5">
              <button
                onClick={() => setIsInvertedMask(!isInvertedMask)}
                className="text-[10px] font-mono text-slate-600 bg-white hover:bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200 flex items-center gap-1 transition-colors"
                title="Invert Binary Mask"
              >
                <Contrast className="w-3 h-3" />
                <span>{isInvertedMask ? 'Normal' : 'Invert'}</span>
              </button>
              <span className="text-[11px] font-mono text-slate-500 bg-white px-2 py-0.5 rounded border border-slate-200">
                Binary (0/1)
              </span>
            </div>
          </div>

          <div className="p-3 bg-black flex-1 flex items-center justify-center min-h-[260px] sm:min-h-[300px] relative group">
            <img
              src={result.mask_base64}
              alt="Predicted Lesion Segmentation Mask"
              className={`max-h-[320px] w-full object-contain rounded-lg transition-transform duration-200 group-hover:scale-[1.01] ${
                isInvertedMask ? 'invert' : ''
              }`}
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/70 backdrop-blur-sm rounded text-[10px] text-teal-300 font-mono flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse"></span>
              <span>BVI-Net Output</span>
            </div>
          </div>

          <div className="p-2.5 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500">
            <span>Threshold: &tau; &ge; 0.50</span>
            <button
              id="download-mask-btn"
              onClick={() => downloadImage(result.mask_base64, `mask_${result.fileName || 'pred.png'}`)}
              className="inline-flex items-center gap-1 text-xs font-semibold text-teal-700 hover:text-teal-800 p-1 rounded hover:bg-teal-50 transition-colors"
              title="Download Predicted Binary Mask PNG"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download PNG</span>
            </button>
          </div>
        </div>

        {/* 3. OVERLAY (Original + Highlighted Lesion) */}
        <div className="bg-white rounded-xl border border-slate-200/90 shadow-xs overflow-hidden flex flex-col">
          <div className="bg-slate-50/90 px-3.5 py-2.5 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  overlayColor === 'red'
                    ? 'bg-red-500'
                    : overlayColor === 'teal'
                    ? 'bg-teal-500'
                    : overlayColor === 'emerald'
                    ? 'bg-emerald-500'
                    : 'bg-amber-500'
                }`}
              ></span>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                3. Clinical Overlay
              </h4>
            </div>
            <span className="text-[11px] font-mono text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-200/60 font-semibold">
              Alpha: {Math.round(overlayOpacity * 100)}%
            </span>
          </div>

          <div className="p-3 bg-slate-950 flex-1 flex items-center justify-center min-h-[260px] sm:min-h-[300px] relative overflow-hidden select-none">
            {/* Split Comparison Slider Mode */}
            {isSplitMode ? (
              <div className="relative w-full h-[280px] flex items-center justify-center overflow-hidden rounded-lg">
                {/* Background: Raw Original Image */}
                <img
                  src={result.original_image_url}
                  alt="Raw Base"
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                />

                {/* Foreground Clipped: Overlay */}
                <div
                  className="absolute inset-0 overflow-hidden"
                  style={{ clipPath: `inset(0 ${100 - splitPosition}% 0 0)` }}
                >
                  <img
                    src={renderedOverlayDataUrl || result.original_image_url}
                    alt="Overlay Top"
                    className="w-full h-full object-contain pointer-events-none"
                  />
                </div>

                {/* Vertical Divider Handle Line */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.8)] z-10 cursor-ew-resize flex items-center justify-center"
                  style={{ left: `${splitPosition}%` }}
                >
                  <div className="w-5 h-5 rounded-full bg-white text-slate-800 shadow-md flex items-center justify-center text-[10px] font-bold">
                    &harr;
                  </div>
                </div>

                {/* Interactive Slider Input */}
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={splitPosition}
                  onChange={(e) => setSplitPosition(parseFloat(e.target.value))}
                  className="absolute inset-0 opacity-0 cursor-ew-resize z-20 w-full h-full"
                />

                <div className="absolute top-2 left-2 px-1.5 py-0.5 bg-black/60 rounded text-[9px] text-white font-mono z-10">
                  Overlay
                </div>
                <div className="absolute top-2 right-2 px-1.5 py-0.5 bg-black/60 rounded text-[9px] text-white font-mono z-10">
                  Original
                </div>
              </div>
            ) : (
              /* Standard High-Quality Overlay Display */
              <div className="relative w-full h-full flex items-center justify-center group">
                <img
                  src={renderedOverlayDataUrl || result.original_image_url}
                  alt="Lesion Segmentation Overlay"
                  className="max-h-[320px] w-full object-contain rounded-lg transition-transform duration-200 group-hover:scale-[1.01]"
                />
                <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/60 backdrop-blur-sm rounded text-[10px] text-slate-200 font-mono">
                  Translucent Mask Map
                </div>
              </div>
            )}
          </div>

          <div className="p-2.5 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500">
            <span className="font-semibold text-slate-700 capitalize">
              {overlayColor} Tint ({Math.round(overlayOpacity * 100)}%)
            </span>
            <button
              id="download-overlay-btn"
              onClick={() => {
                if (renderedOverlayDataUrl) {
                  downloadImage(renderedOverlayDataUrl, `overlay_${result.fileName || 'composite.png'}`);
                }
              }}
              className="inline-flex items-center gap-1 text-xs font-semibold text-teal-700 hover:text-teal-800 p-1 rounded hover:bg-teal-50 transition-colors"
              title="Download High-Res Overlay Composite PNG"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Composite</span>
            </button>
          </div>
        </div>
      </div>

      {/* Hidden processing canvas */}
      <canvas ref={overlayCanvasRef} className="hidden" />
    </div>
  );
};
