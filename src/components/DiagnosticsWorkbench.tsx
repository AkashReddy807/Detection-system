import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  Download,
  Eye,
  FileImage,
  Lock,
  RefreshCw,
  Sliders,
  Upload,
} from 'lucide-react';
import { SAMPLE_SPECIMENS } from '../data/sampleSpecimens';
import { PLANT_VILLAGE_CLASS_MAP } from '../data/plantVillage';
import type { LeafAnalysisResult } from '../types/diagnostics';
import {
  buildResNet224Preview,
  computeGradCamField,
  renderGradCamToCanvas,
  sampleFieldAt,
  type GradCamField,
} from '../utils/gradcam';

type ViewMode = 'side-by-side' | 'curtain' | 'heatmap-only';

interface DiagnosticsWorkbenchProps {
  activeImageSrc: string | null;
  activeSpecimenLabel: string;
  analysis: LeafAnalysisResult | null;
  isAnalyzing: boolean;
  error: string | null;
  onSelectImage: (imageSrc: string, label: string, autoAnalyze?: boolean) => void;
  onRunAnalysis: () => void;
  onGradCamExportReady: (dataUrl: string | null) => void;
  onOpenPhase2: () => void;
}

export const DiagnosticsWorkbench: React.FC<DiagnosticsWorkbenchProps> = ({
  activeImageSrc,
  activeSpecimenLabel,
  analysis,
  isAnalyzing,
  error,
  onSelectImage,
  onRunAnalysis,
  onGradCamExportReady,
  onOpenPhase2,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');
  const [alpha, setAlpha] = useState<number>(0.52);
  const [threshold, setThreshold] = useState<number>(0.18);
  const [showBoxes, setShowBoxes] = useState<boolean>(false);
  const [curtainPos, setCurtainPos] = useState<number>(50);
  const [isDraggingFile, setIsDraggingFile] = useState<boolean>(false);
  const [preview224Url, setPreview224Url] = useState<string | null>(null);
  const [imgDims, setImgDims] = useState<{ w: number; h: number } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [loadedImg, setLoadedImg] = useState<HTMLImageElement | null>(null);
  const [probe, setProbe] = useState<{ x: number; y: number; value: number } | null>(
    null
  );

  const gradCamFieldRef = useRef<GradCamField | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Load image into HTMLImageElement & build 224x224 preview
  useEffect(() => {
    if (!activeImageSrc) {
      setLoadedImg(null);
      gradCamFieldRef.current = null;
      setPreview224Url(null);
      setImgDims(null);
      return;
    }

    let cancelled = false;
    const img = new Image();
    if (/^https?:\/\//i.test(activeImageSrc)) {
      img.crossOrigin = 'anonymous';
    }
    img.onload = () => {
      if (cancelled) return;
      setLoadedImg(img);
      setImgDims({ w: img.naturalWidth, h: img.naturalHeight });
      setPreview224Url(buildResNet224Preview(img));
    };
    img.onerror = () => {
      if (cancelled) return;
      setLoadedImg(null);
      setUploadError('Could not decode the selected image file. Please try a standard JPG or PNG leaf photo.');
    };
    img.src = activeImageSrc;

    return () => {
      cancelled = true;
    };
  }, [activeImageSrc]);

  // Recompute Grad-CAM field whenever loaded image or analysis updates
  useEffect(() => {
    if (!loadedImg || !analysis) {
      gradCamFieldRef.current = null;
      onGradCamExportReady(null);
      return;
    }
    const field = computeGradCamField(
      loadedImg,
      analysis.lesionRegions,
      analysis.isHealthy,
      112
    );
    gradCamFieldRef.current = field;
    if (overlayCanvasRef.current) {
      renderGradCamToCanvas(overlayCanvasRef.current, loadedImg, field, {
        alpha,
        threshold,
        heatmapOnly: viewMode === 'heatmap-only',
        showBoundingBoxes: showBoxes,
        lesionRegions: analysis.lesionRegions,
      });
      onGradCamExportReady(overlayCanvasRef.current.toDataURL('image/png'));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedImg, analysis]);

  // Re-render Grad-CAM overlay whenever calibration sliders or view mode change
  useEffect(() => {
    const field = gradCamFieldRef.current;
    const canvas = overlayCanvasRef.current;
    if (!loadedImg || !field || !canvas || !analysis) return;

    renderGradCamToCanvas(canvas, loadedImg, field, {
      alpha,
      threshold,
      heatmapOnly: viewMode === 'heatmap-only',
      showBoundingBoxes: showBoxes,
      lesionRegions: analysis.lesionRegions,
    });
    onGradCamExportReady(canvas.toDataURL('image/png'));
  }, [alpha, threshold, showBoxes, viewMode, loadedImg, analysis, onGradCamExportReady]);

  const handleFile = useCallback(
    (file: File) => {
      setUploadError(null);
      const isImageType =
        (file.type && file.type.startsWith('image/')) ||
        /\.(png|jpe?g|webp|gif|bmp|avif|jfif)$/i.test(file.name);

      if (!isImageType) {
        setUploadError(
          `Unsupported file "${file.name}". Please upload a PNG, JPG, JPEG, or WebP leaf image.`
        );
        return;
      }

      if (file.size > 20 * 1024 * 1024) {
        setUploadError('Image exceeds 20 MB limit. Please select a smaller leaf photograph.');
        return;
      }

      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          onSelectImage(reader.result, file.name || 'Uploaded leaf image', true);
        } else {
          setUploadError('Failed to read image data. Please try another file.');
        }
      };
      reader.onerror = () => {
        setUploadError('Unable to read the selected file from disk. Please try again.');
      };
      reader.readAsDataURL(file);
    },
    [onSelectImage]
  );

  // Support pasting a leaf image directly from clipboard (Ctrl+V / Cmd+V)
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const blob = items[i].getAsFile();
          if (blob) {
            e.preventDefault();
            handleFile(blob);
            break;
          }
        }
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [handleFile]);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFile(false);
    const file =
      e.dataTransfer.files?.[0] ||
      (e.dataTransfer.items?.[0]?.kind === 'file'
        ? e.dataTransfer.items[0].getAsFile()
        : null);
    if (file) {
      handleFile(file);
    } else {
      setUploadError('No image file detected in drop. Please select a file using the browse button.');
    }
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const xNorm = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const yNorm = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    const val = sampleFieldAt(gradCamFieldRef.current, xNorm, yNorm);
    setProbe({
      x: Math.round(xNorm * 224),
      y: Math.round(yNorm * 224),
      value: val,
    });
  };

  const handleExportPng = () => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    const safeName = (analysis?.predictedClass || 'leaf').replace(/[^a-z0-9_-]/gi, '_');
    link.download = `gradcam_${safeName}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const conditionType = analysis?.isHealthy ? 'healthy' : 'diseased';
  const phase2PromptPreview = analysis
    ? `You are an expert plant pathologist.\nThe model detected that the plant is "${analysis.readablePrediction}", which means it is ${conditionType}.\nGive concise, detailed points of actionable treatment and preventive care advice for this condition.`
    : `You are an expert plant pathologist.\nThe model detected that the plant is "{readable_prediction}", which means it is {condition_type}.\nGive concise, detailed points of actionable treatment and preventive care advice for this condition.`;

  return (
    <div className="max-w-[1440px] mx-auto px-6 py-8">
      {/* Section Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-6 border-b border-slate-200">
        <div>
          <p className="text-xs font-mono text-slate-500 tracking-wide mb-1.5">
            Semester 1 Scope · ResNet50 Classification &amp; Grad-CAM Explainability
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
            Leaf Pathology Diagnostic Console
          </h1>
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-500 font-mono tabular-nums">
          <span>Input 224×224 RGB</span>
          <span aria-hidden="true">·</span>
          <span>38 PlantVillage Classes</span>
          <span aria-hidden="true">·</span>
          <span>Layer conv5_block3_out</span>
        </div>
      </div>

      {/* Asymmetric Split Console */}
      <div className="mt-6 grid grid-cols-1 lg:grid-cols-[340px_minmax(0,1fr)] gap-6 items-start">
        {/* LEFT COLUMN: Specimen Intake & Grad-CAM Calibration */}
        <aside className="space-y-6">
          {/* Upload Panel */}
          <section className="bg-white border border-slate-200 rounded-lg p-5">
            <h2 className="text-sm font-semibold text-slate-900 mb-1">
              1. Leaf Specimen Input
            </h2>
            <p className="text-xs text-slate-500 mb-4">
              Upload a clear leaf photograph (JPG or PNG) or select a laboratory reference specimen.
            </p>

            <div
              onDragEnter={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDraggingFile(true);
              }}
              onDragOver={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDraggingFile(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setIsDraggingFile(false);
              }}
              onDrop={handleDrop}
              className={`relative rounded-md border border-dashed p-5 text-center transition-colors focus-within:ring-2 focus-within:ring-teal-600 ${
                isDraggingFile
                  ? 'border-teal-600 bg-teal-50/50'
                  : 'border-slate-300 bg-slate-50 hover:border-slate-400 hover:bg-slate-100/60'
              }`}
            >
              <input
                id="leaf-file-upload"
                type="file"
                accept="image/*"
                aria-label="Upload leaf image"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                onChange={(e) => {
                  const input = e.currentTarget;
                  const f = input.files?.[0];
                  if (f) {
                    handleFile(f);
                  }
                  input.value = '';
                }}
              />
              <Upload className="w-5 h-5 mx-auto text-slate-500 mb-2 pointer-events-none" />
              <p className="text-sm font-medium text-slate-800 pointer-events-none">
                Drop leaf image, paste, or browse
              </p>
              <p className="text-xs text-slate-500 mt-1 pointer-events-none">
                PNG, JPG, WebP · Auto-resized to 224×224
              </p>
              <span className="mt-3 inline-flex items-center justify-center px-3.5 py-1.5 text-xs font-medium text-slate-800 bg-white border border-slate-300 rounded-md shadow-2xs pointer-events-none">
                Choose Leaf Image
              </span>
            </div>

            {uploadError && (
              <p className="mt-2.5 text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-md px-3 py-2">
                {uploadError}
              </p>
            )}

            {/* Quick-Load Lab Reference Specimens */}
            <div className="mt-5">
              <div className="flex items-center justify-between mb-2.5">
                <span className="text-xs font-medium text-slate-700">
                  Reference Lab Specimens
                </span>
                <span className="text-xs font-mono text-slate-400">4 samples</span>
              </div>
              <div className="grid grid-cols-2 gap-2.5">
                {SAMPLE_SPECIMENS.map((spec) => {
                  const isActive = activeImageSrc === spec.imageUrl;
                  return (
                    <button
                      key={spec.id}
                      type="button"
                      onClick={() => onSelectImage(spec.imageUrl, spec.label, true)}
                      className={`group text-left rounded-md border p-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 ${
                        isActive
                          ? 'border-teal-700 bg-teal-50/40'
                          : 'border-slate-200 bg-white hover:border-slate-300'
                      }`}
                    >
                      <div className="aspect-square w-full overflow-hidden rounded bg-slate-100 mb-2 flex items-center justify-center">
                        <img
                          src={spec.imageUrl}
                          alt={spec.label}
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <p className="text-xs font-medium text-slate-900 truncate">
                        {spec.label}
                      </p>
                      <p className="text-[11px] font-mono text-slate-500 truncate">
                        {spec.specimenCode}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>

            <button
              type="button"
              onClick={onRunAnalysis}
              disabled={!activeImageSrc || isAnalyzing}
              className="mt-5 w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium text-white bg-teal-700 rounded-md hover:bg-teal-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
            >
              <RefreshCw className={`w-4 h-4 ${isAnalyzing ? 'animate-spin' : ''}`} />
              {isAnalyzing ? 'Running Inference...' : analysis ? 'Re-run Diagnosis' : 'Run Diagnosis'}
            </button>
          </section>

          {/* 224x224 Preprocessing Tensor Stage */}
          <section className="bg-white border border-slate-200 rounded-lg p-5">
            <h2 className="text-sm font-semibold text-slate-900 mb-3">
              2. Preprocessing Tensor (224×224)
            </h2>
            <div className="flex items-center gap-4">
              <div className="w-20 h-20 rounded border border-slate-200 bg-slate-50 overflow-hidden shrink-0 flex items-center justify-center">
                {preview224Url ? (
                  <img
                    src={preview224Url}
                    alt="224x224 resized ResNet50 input"
                    referrerPolicy="no-referrer"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <FileImage className="w-6 h-6 text-slate-300" />
                )}
              </div>
              <dl className="text-xs space-y-1.5 flex-1 min-w-0 font-mono tabular-nums">
                <div className="flex justify-between gap-2">
                  <dt className="text-slate-500 font-sans">Source</dt>
                  <dd className="text-slate-800 truncate">
                    {imgDims ? `${imgDims.w}×${imgDims.h}px` : '—'}
                  </dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-slate-500 font-sans">Tensor</dt>
                  <dd className="text-slate-800">(1, 224, 224, 3)</dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt className="text-slate-500 font-sans">Norm</dt>
                  <dd className="text-slate-800">ImageNet BGR</dd>
                </div>
              </dl>
            </div>
          </section>

          {/* Grad-CAM Calibration Controls */}
          <section className="bg-white border border-slate-200 rounded-lg p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-slate-500" />
                3. Grad-CAM Calibration
              </h2>
              <button
                type="button"
                onClick={() => {
                  setAlpha(0.52);
                  setThreshold(0.18);
                  setShowBoxes(false);
                }}
                className="text-xs text-teal-700 hover:underline whitespace-nowrap"
              >
                Reset
              </button>
            </div>

            <div className="space-y-4">
              {/* Overlay Opacity */}
              <div>
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <label htmlFor="alpha-slider" className="text-slate-700 font-medium">
                    Heatmap Overlay Alpha
                  </label>
                  <span className="font-mono tabular-nums text-slate-600">
                    {Math.round(alpha * 100)}%
                  </span>
                </div>
                <input
                  id="alpha-slider"
                  type="range"
                  min={0.15}
                  max={0.9}
                  step={0.02}
                  value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="lab-slider w-full"
                />
              </div>

              {/* Activation Threshold */}
              <div>
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <label htmlFor="thresh-slider" className="text-slate-700 font-medium">
                    Activation Cutoff (τ)
                  </label>
                  <span className="font-mono tabular-nums text-slate-600">
                    {threshold.toFixed(2)}
                  </span>
                </div>
                <input
                  id="thresh-slider"
                  type="range"
                  min={0}
                  max={0.7}
                  step={0.02}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="lab-slider w-full"
                />
              </div>

              {/* Lesion Boxes Toggle */}
              <label className="flex items-center justify-between pt-2 border-t border-slate-100 cursor-pointer">
                <span className="text-xs font-medium text-slate-700">
                  Show Focal Lesion Regions
                </span>
                <input
                  type="checkbox"
                  checked={showBoxes}
                  onChange={(e) => setShowBoxes(e.target.checked)}
                  className="h-4 w-4 rounded border-slate-300 text-teal-700 focus:ring-teal-600"
                />
              </label>
            </div>
          </section>
        </aside>

        {/* RIGHT COLUMN: Primary Diagnosis Stage & Grad-CAM Viewport */}
        <div className="space-y-6 min-w-0">
          {/* Error Banner */}
          {error && (
            <div className="bg-rose-50 border border-rose-200 rounded-lg p-4 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-rose-700 shrink-0 mt-0.5" />
              <div className="min-w-0">
                <p className="text-sm font-semibold text-rose-900">
                  Inference request could not be completed
                </p>
                <p className="text-xs text-rose-800 mt-1">{error}</p>
              </div>
            </div>
          )}

          {/* Diagnosis Summary Strip */}
          <section className="bg-white border border-slate-200 rounded-lg p-6">
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 pb-5 border-b border-slate-200">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500 mb-1.5">
                  <span>Specimen: {activeSpecimenLabel || 'None selected'}</span>
                  {analysis && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span>Crop: {analysis.crop}</span>
                      <span aria-hidden="true">·</span>
                      <span
                        className={`font-medium ${
                          analysis.isHealthy ? 'text-emerald-700' : 'text-rose-700'
                        }`}
                      >
                        {analysis.isHealthy ? '● Healthy' : '▲ Diseased'}
                      </span>
                    </>
                  )}
                </div>
                <h2 className="text-xl sm:text-2xl font-semibold text-slate-900 tracking-tight">
                  {isAnalyzing
                    ? 'Analyzing leaf pathology and spatial activations...'
                    : analysis
                    ? analysis.readablePrediction
                    : 'Select or upload a leaf image to begin'}
                </h2>
                {analysis && (
                  <p className="text-xs text-slate-500 mt-1 font-mono">
                    Class ID: {analysis.predictedClass} · Agent: {analysis.pathogen}
                  </p>
                )}
              </div>

              {/* Quantitative Metrics */}
              <div className="flex items-center gap-6 shrink-0">
                <div>
                  <p className="text-[11px] font-medium text-slate-500">Confidence</p>
                  <p className="text-2xl font-mono font-semibold tabular-nums text-slate-900">
                    {analysis ? `${(analysis.confidence * 100).toFixed(2)}%` : '—'}
                  </p>
                </div>
                <div className="h-9 w-px bg-slate-200" aria-hidden="true" />
                <div>
                  <p className="text-[11px] font-medium text-slate-500">Affected Area</p>
                  <p className="text-2xl font-mono font-semibold tabular-nums text-slate-900">
                    {analysis ? `${analysis.leafCoverageEstimate}%` : '—'}
                  </p>
                </div>
                <div className="h-9 w-px bg-slate-200" aria-hidden="true" />
                <div>
                  <p className="text-[11px] font-medium text-slate-500">Focal Regions</p>
                  <p className="text-2xl font-mono font-semibold tabular-nums text-slate-900">
                    {analysis ? analysis.lesionRegions.length : '—'}
                  </p>
                </div>
              </div>
            </div>

            {/* Viewport Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 py-4">
              <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
                {(
                  [
                    { id: 'side-by-side', label: 'Side-by-Side' },
                    { id: 'curtain', label: 'Split Curtain' },
                    { id: 'heatmap-only', label: 'Heatmap Only' },
                  ] as const
                ).map((mode) => (
                  <button
                    key={mode.id}
                    type="button"
                    onClick={() => setViewMode(mode.id)}
                    className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                      viewMode === mode.id
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-4">
                {probe && analysis && (
                  <span className="text-xs font-mono tabular-nums text-slate-600">
                    Cursor ({probe.x}, {probe.y}) · α = {probe.value.toFixed(2)}
                  </span>
                )}
                <button
                  type="button"
                  onClick={handleExportPng}
                  disabled={!analysis}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                >
                  <Download className="w-3.5 h-3.5" />
                  Export Grad-CAM PNG
                </button>
              </div>
            </div>

            {/* Grad-CAM Visual Canvas Area */}
            {!activeImageSrc ? (
              <div className="h-[380px] rounded-md border border-dashed border-slate-200 bg-slate-50 flex flex-col items-center justify-center text-center p-8">
                <Eye className="w-8 h-8 text-slate-300 mb-3" />
                <p className="text-sm font-medium text-slate-700">
                  No specimen loaded in viewport
                </p>
                <p className="text-xs text-slate-500 mt-1 max-w-sm">
                  Upload a leaf image or click one of the 4 reference PlantVillage specimens on the left to inspect ResNet50 predictions and Grad-CAM lesion heatmaps.
                </p>
              </div>
            ) : viewMode === 'curtain' ? (
              <div className="space-y-3">
                <div className="relative max-w-[512px] mx-auto aspect-square rounded-md overflow-hidden border border-slate-200 bg-slate-100 select-none">
                  {/* Base: Grad-CAM Canvas */}
                  <canvas
                    ref={overlayCanvasRef}
                    onMouseMove={handleCanvasMove}
                    onMouseLeave={() => setProbe(null)}
                    className="w-full h-full object-contain cursor-crosshair block"
                  />
                  {/* Clipped Original Leaf on the Left */}
                  <div
                    className="absolute inset-0 pointer-events-none overflow-hidden"
                    style={{ clipPath: `inset(0 ${100 - curtainPos}% 0 0)` }}
                  >
                    <img
                      src={activeImageSrc}
                      alt="Original leaf specimen"
                      referrerPolicy="no-referrer"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  {/* Divider Line */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-white shadow pointer-events-none"
                    style={{ left: `${curtainPos}%` }}
                  />
                  <span className="absolute top-2.5 left-2.5 px-2 py-0.5 text-[11px] font-mono bg-slate-900/75 text-white rounded">
                    Original
                  </span>
                  <span className="absolute top-2.5 right-2.5 px-2 py-0.5 text-[11px] font-mono bg-slate-900/75 text-white rounded">
                    Grad-CAM
                  </span>
                </div>
                <div className="max-w-[512px] mx-auto flex items-center gap-3">
                  <span className="text-xs font-mono text-slate-500">Original</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={curtainPos}
                    onChange={(e) => setCurtainPos(Number(e.target.value))}
                    aria-label="Split curtain comparison slider"
                    className="lab-slider flex-1"
                  />
                  <span className="text-xs font-mono text-slate-500">Grad-CAM</span>
                </div>
              </div>
            ) : (
              <div
                className={`grid gap-5 ${
                  viewMode === 'heatmap-only'
                    ? 'grid-cols-1 max-w-[512px] mx-auto'
                    : 'grid-cols-1 md:grid-cols-2'
                }`}
              >
                {viewMode === 'side-by-side' && (
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-medium text-slate-700">
                        Original RGB Specimen
                      </span>
                      <span className="text-[11px] font-mono text-slate-400">
                        Reference
                      </span>
                    </div>
                    <div className="aspect-square rounded-md overflow-hidden border border-slate-200 bg-slate-100">
                      <img
                        src={activeImageSrc}
                        alt={activeSpecimenLabel}
                        referrerPolicy="no-referrer"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                )}

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-slate-700">
                      {viewMode === 'heatmap-only'
                        ? 'Grad-CAM Activation Map (COLORMAP_JET)'
                        : 'Grad-CAM Lesion Overlay (conv5_block3_out)'}
                    </span>
                    <span className="text-[11px] font-mono text-slate-400">
                      {isAnalyzing ? 'Computing...' : 'Hover to probe'}
                    </span>
                  </div>
                  <div className="relative aspect-square rounded-md overflow-hidden border border-slate-200 bg-slate-100">
                    <canvas
                      ref={overlayCanvasRef}
                      onMouseMove={handleCanvasMove}
                      onMouseLeave={() => setProbe(null)}
                      className="w-full h-full object-cover cursor-crosshair block"
                    />
                    {isAnalyzing && (
                      <div className="absolute inset-0 bg-white/75 flex flex-col items-center justify-center gap-2">
                        <RefreshCw className="w-5 h-5 text-teal-700 animate-spin" />
                        <span className="text-xs font-mono text-slate-700">
                          Computing ResNet50 &amp; Grad-CAM map...
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Calibrated COLORMAP_JET Scale Legend */}
            <div className="mt-5 pt-4 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <span className="text-xs text-slate-600">
                Grad-CAM Activation Intensity (COLORMAP_JET)
              </span>
              <div className="flex items-center gap-2.5">
                <span className="text-[11px] font-mono tabular-nums text-slate-500">
                  0.00 Low
                </span>
                <div
                  className="w-44 h-2.5 rounded-xs border border-slate-200"
                  style={{
                    background:
                      'linear-gradient(90deg, #00008F 0%, #0000FF 15%, #00FFFF 38%, #00FF00 55%, #FFFF00 78%, #FF0000 100%)',
                  }}
                />
                <span className="text-[11px] font-mono tabular-nums text-slate-500">
                  1.00 Peak Lesion
                </span>
              </div>
            </div>
          </section>

          {/* Bottom Split: Top-5 Softmax Table + Visual Pathology Notes */}
          {analysis && (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {/* Softmax Distribution */}
              <section className="bg-white border border-slate-200 rounded-lg p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-slate-900">
                    Top-5 Softmax Class Distribution
                  </h3>
                  <span className="text-xs font-mono text-slate-400">Dense(38)</span>
                </div>
                <div className="space-y-3">
                  {analysis.topPredictions.map((pred, idx) => {
                    const info = PLANT_VILLAGE_CLASS_MAP[pred.classId];
                    const pct = Math.max(0, Math.min(100, pred.probability * 100));
                    return (
                      <div key={`${pred.classId}-${idx}`}>
                        <div className="flex items-baseline justify-between gap-2 text-xs mb-1">
                          <span
                            className={`truncate ${
                              idx === 0
                                ? 'font-semibold text-slate-900'
                                : 'text-slate-600'
                            }`}
                          >
                            {info
                              ? `${info.crop} — ${info.condition}`
                              : pred.classId}
                          </span>
                          <span className="font-mono tabular-nums text-slate-700 shrink-0">
                            {pct.toFixed(2)}%
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-100 rounded-xs overflow-hidden">
                          <div
                            className={`h-full rounded-xs ${
                              idx === 0 ? 'bg-teal-700' : 'bg-slate-400'
                            }`}
                            style={{ width: `${Math.max(1.5, pct)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>

              {/* Observed Morphological Indicators */}
              <section className="bg-white border border-slate-200 rounded-lg p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-slate-900">
                    Observed Visual Pathology Features
                  </h3>
                  <span className="text-xs font-mono text-slate-400">
                    Phase 1 Vision Output
                  </span>
                </div>
                <ul className="space-y-2.5 text-sm text-slate-700 leading-relaxed">
                  {analysis.visualObservations.map((obs, i) => (
                    <li key={i} className="flex items-start gap-2.5">
                      <span className="font-mono text-xs text-slate-400 mt-0.5 tabular-nums">
                        0{i + 1}.
                      </span>
                      <span>{obs}</span>
                    </li>
                  ))}
                </ul>
                {analysis.lesionRegions.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-100 text-xs text-slate-500">
                    Localized focal features:{' '}
                    <span className="text-slate-700">
                      {Array.from(
                        new Set(analysis.lesionRegions.map((r) => r.label))
                      )
                        .slice(0, 4)
                        .join(' · ')}
                    </span>
                  </div>
                )}
              </section>
            </div>
          )}

          {/* Phase 2 Handoff Preview: LLM Guidance Scheduled for Next Semester */}
          <section className="bg-slate-50 border border-slate-200 rounded-lg p-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
              <div className="flex items-center gap-2.5">
                <Lock className="w-4 h-4 text-amber-700 shrink-0" />
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    LLM Plant Care Guidance Engine — Scheduled for Semester 2 (Phase 2)
                  </h3>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Phase 1 outputs the verified disease class and lesion context below. Live Mistral-7B-Instruct recommendation synthesis will be connected next semester.
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={onOpenPhase2}
                className="px-3.5 py-2 text-xs font-medium text-slate-800 bg-white border border-slate-300 rounded-md hover:bg-slate-100 transition-colors whitespace-nowrap shrink-0"
              >
                View Phase 2 Plan
              </button>
            </div>

            <div className="mt-4 grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-5">
              <div>
                <p className="text-xs font-medium text-slate-700 mb-2">
                  Auto-Populated Prompt Handoff Payload (Ready for Phase 2 API)
                </p>
                <pre className="p-3.5 bg-white border border-slate-200 rounded-md text-xs font-mono text-slate-700 whitespace-pre-wrap leading-relaxed overflow-x-auto">
                  {phase2PromptPreview}
                </pre>
              </div>
              <div className="bg-white border border-slate-200 rounded-md p-4 text-xs space-y-2.5">
                <p className="font-semibold text-slate-900">Phase 2 Target Integration</p>
                <div className="flex justify-between gap-2 text-slate-600">
                  <span>Target Model</span>
                  <span className="font-mono text-slate-900">Mistral-7B-Instruct</span>
                </div>
                <div className="flex justify-between gap-2 text-slate-600">
                  <span>Inference API</span>
                  <span className="font-mono text-slate-900">Hugging Face Hub</span>
                </div>
                <div className="flex justify-between gap-2 text-slate-600">
                  <span>Status</span>
                  <span className="font-medium text-amber-700">○ Next Semester</span>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
