import { useCallback, useEffect, useRef, useState } from 'react';
import { DiagnosticsWorkbench } from './components/DiagnosticsWorkbench';
import { ModelLab } from './components/ModelLab';
import { PhaseRoadmap } from './components/PhaseRoadmap';
import { AcademicReport } from './components/AcademicReport';
import { SAMPLE_SPECIMENS } from './data/sampleSpecimens';
import type { LeafAnalysisResult, ReportMetadata } from './types/diagnostics';

type ActiveTab = 'diagnostics' | 'model-lab' | 'roadmap' | 'report';

async function imageUrlToBase64(
  src: string
): Promise<{ base64: string; mimeType: string; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    if (/^https?:\/\//i.test(src)) {
      img.crossOrigin = 'anonymous';
    }
    img.onload = () => {
      const maxDim = 768;
      const scale = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.max(1, Math.round(img.naturalWidth * scale));
      const h = Math.max(1, Math.round(img.naturalHeight * scale));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Could not initialize image canvas context.'));
        return;
      }
      ctx.drawImage(img, 0, 0, w, h);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
      const base64 = dataUrl.split(',')[1] ?? '';
      resolve({
        base64,
        mimeType: 'image/jpeg',
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    };
    img.onerror = () => reject(new Error('Failed to load selected leaf image.'));
    img.src = src;
  });
}

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('diagnostics');
  const [activeImageSrc, setActiveImageSrc] = useState<string | null>(
    SAMPLE_SPECIMENS[0].imageUrl
  );
  const [activeSpecimenLabel, setActiveSpecimenLabel] = useState<string>(
    SAMPLE_SPECIMENS[0].label
  );
  const [analysis, setAnalysis] = useState<LeafAnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [gradCamSnapshotUrl, setGradCamSnapshotUrl] = useState<string | null>(null);

  const [reportMeta, setReportMeta] = useState<ReportMetadata>({
    studentName: 'Akash Reddy Maligireddy',
    rollNumber: 'CS-2026-041',
    facultyGuide: 'Dr. Faculty Supervisor',
    department: 'Department of Computer Science & Engineering',
    institution: 'School of Engineering & Technology',
    academicTerm: 'Semester 1 · Phase 1 (2026–2027)',
  });

  const requestCounterRef = useRef(0);

  const runDiagnosisForImage = useCallback(async (src: string) => {
    const reqId = ++requestCounterRef.current;
    setIsAnalyzing(true);
    setError(null);

    try {
      const { base64, mimeType, width, height } = await imageUrlToBase64(src);
      const response = await fetch('/api/analyze-leaf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageBase64: base64,
          mimeType,
          imageWidth: width,
          imageHeight: height,
        }),
      });

      const payload = await response.json();
      if (reqId !== requestCounterRef.current) return;

      if (!response.ok) {
        throw new Error(payload?.error || `Server responded with ${response.status}`);
      }

      setAnalysis(payload as LeafAnalysisResult);
    } catch (err: unknown) {
      if (reqId !== requestCounterRef.current) return;
      const msg =
        err instanceof Error ? err.message : 'Unable to complete leaf diagnosis.';
      setError(msg);
      setAnalysis(null);
    } finally {
      if (reqId === requestCounterRef.current) {
        setIsAnalyzing(false);
      }
    }
  }, []);

  // Automatically run initial diagnosis on the default lab reference specimen
  useEffect(() => {
    runDiagnosisForImage(SAMPLE_SPECIMENS[0].imageUrl);
  }, [runDiagnosisForImage]);

  const handleSelectImage = useCallback(
    (imageSrc: string, label: string, autoAnalyze = true) => {
      setActiveImageSrc(imageSrc);
      setActiveSpecimenLabel(label);
      setAnalysis(null);
      setGradCamSnapshotUrl(null);
      if (autoAnalyze) {
        runDiagnosisForImage(imageSrc);
      }
    },
    [runDiagnosisForImage]
  );

  const handleRunCurrentAnalysis = useCallback(() => {
    if (activeImageSrc) {
      runDiagnosisForImage(activeImageSrc);
    }
  }, [activeImageSrc, runDiagnosisForImage]);

  const handleGradCamExportReady = useCallback((dataUrl: string | null) => {
    setGradCamSnapshotUrl(dataUrl);
  }, []);

  const navItems: Array<{ id: ActiveTab; label: string }> = [
    { id: 'diagnostics', label: 'Diagnostics' },
    { id: 'model-lab', label: 'Model Lab' },
    { id: 'roadmap', label: 'Phase 2 Plan' },
    { id: 'report', label: 'Phase 1 Report' },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-[#F8FAFC] text-slate-900">
      {/* Strict 3-Zone Top Navigation Bar */}
      <header className="no-print sticky top-0 z-30 bg-white border-b border-slate-200">
        <div className="max-w-[1440px] mx-auto px-6 h-16 flex items-center justify-between gap-8">
          {/* Zone 1: Single-element brand wordmark */}
          <button
            type="button"
            onClick={() => setActiveTab('diagnostics')}
            className="text-lg font-bold tracking-tight text-slate-950 whitespace-nowrap shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 rounded"
          >
            PhytoScan Lab
          </button>

          {/* Zone 2: 4 concise single-line text links */}
          <nav
            aria-label="Primary sections"
            className="flex items-center gap-6 overflow-x-auto"
          >
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  className={`py-1 text-sm font-medium transition-colors whitespace-nowrap shrink-0 border-b-2 ${
                    isActive
                      ? 'text-teal-800 border-teal-700'
                      : 'text-slate-600 border-transparent hover:text-slate-950 hover:border-slate-300'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: 1 Primary Action */}
          <div className="hidden sm:flex items-center shrink-0">
            <button
              type="button"
              onClick={() => setActiveTab('report')}
              className="px-4 py-2 text-xs font-medium text-white bg-teal-700 rounded-md hover:bg-teal-800 transition-colors whitespace-nowrap shrink-0"
            >
              Export Report
            </button>
          </div>
        </div>
      </header>

      {/* Active Workspace View */}
      <main className="flex-1">
        {activeTab === 'diagnostics' && (
          <DiagnosticsWorkbench
            activeImageSrc={activeImageSrc}
            activeSpecimenLabel={activeSpecimenLabel}
            analysis={analysis}
            isAnalyzing={isAnalyzing}
            error={error}
            onSelectImage={handleSelectImage}
            onRunAnalysis={handleRunCurrentAnalysis}
            onGradCamExportReady={handleGradCamExportReady}
            onOpenPhase2={() => setActiveTab('roadmap')}
          />
        )}

        {activeTab === 'model-lab' && <ModelLab />}

        {activeTab === 'roadmap' && (
          <PhaseRoadmap
            onOpenReport={() => setActiveTab('report')}
            onOpenDiagnostics={() => setActiveTab('diagnostics')}
          />
        )}

        {activeTab === 'report' && (
          <AcademicReport
            reportMeta={reportMeta}
            onUpdateMeta={setReportMeta}
            activeImageSrc={activeImageSrc}
            activeSpecimenLabel={activeSpecimenLabel}
            gradCamSnapshotUrl={gradCamSnapshotUrl}
            analysis={analysis}
          />
        )}
      </main>

      {/* Quiet Footer (Hidden on Print) */}
      <footer className="no-print border-t border-slate-200 bg-white mt-12">
        <div className="max-w-[1440px] mx-auto px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
          <span>
            PhytoScan Lab · Automated Plant Disease Detection (Phase 1 — ResNet50 &amp; Grad-CAM)
          </span>
          <span className="font-mono">
            PlantVillage 38-Class Benchmark · Semester 2 LLM Guidance Planned
          </span>
        </div>
      </footer>
    </div>
  );
}
