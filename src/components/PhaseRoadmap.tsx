import React from 'react';
import { ArrowRight, Check, Clock } from 'lucide-react';

interface PhaseRoadmapProps {
  onOpenReport: () => void;
  onOpenDiagnostics: () => void;
}

const PHASE_1_MODULES = [
  {
    code: 'M1',
    title: 'PlantVillage Dataset Curation & Preprocessing',
    detail:
      'Stratified 70/15/15 split across ~54,305 RGB images spanning 38 crop-disease categories. Automated 224×224 resizing, ImageNet channel normalization, and flip/rotation/zoom/contrast augmentation.',
    deliverable: 'Validated input tensor pipeline (224×224×3)',
  },
  {
    code: 'M2',
    title: 'Two-Stage ResNet50 Transfer Learning Classifier',
    detail:
      'ImageNet-pretrained ResNet50 backbone paired with GlobalAveragePooling2D, Dense(256, ReLU), Dropout(0.4), and 38-class Softmax head. Fine-tuned last 30 convolutional layers at lr=1e-5 to reach 94% test accuracy.',
    deliverable: 'Trained classifier achieving 94.0% test accuracy (+15% over CNN baseline)',
  },
  {
    code: 'M3',
    title: 'Grad-CAM Explainable AI Localization Engine',
    detail:
      'Computes class-discriminative spatial gradients at conv5_block3_out (7×7×2048), pools feature weights, applies ReLU normalization, and renders COLORMAP_JET overlays with adjustable alpha & threshold controls.',
    deliverable: 'Interactive visual explanation viewport & PNG export',
  },
  {
    code: 'M4',
    title: 'Phase 1 Technical Evaluation & Semester Report',
    detail:
      'Comprehensive quantitative benchmarking against custom CNN baseline, precision/recall/F1 verification, and publication-ready Semester 1 project documentation with live diagnostic case study export.',
    deliverable: 'Printable PDF & Markdown Semester 1 Project Report',
  },
];

const PHASE_2_MODULES = [
  {
    code: 'M5',
    title: 'LLM-Based Plant Health Guidance (Mistral-7B-Instruct)',
    detail:
      'Connects Hugging Face Inference API (mistralai/Mistral-7B-Instruct-v0.2) to dynamically transform predicted disease classes and lesion severity into tailored treatment protocols, preventive best practices, and agronomic care advice.',
    deliverable: 'Live contextual plant pathologist recommendation module',
  },
  {
    code: 'M6',
    title: 'Confidence Calibration & Uncertainty Flagging',
    detail:
      'Implements post-hoc Temperature Scaling and out-of-distribution (OOD) entropy detection to curb overconfidence on non-leaf inputs, blurry photographs, or unseen field crops.',
    deliverable: 'Calibrated ECE confidence indicator & low-certainty rejection guard',
  },
  {
    code: 'M7',
    title: 'Lesion Severity Scoring & Treatment Roadmap',
    detail:
      'Quantifies infected lamina percentage from Grad-CAM activation masks into mild, moderate, and severe clinical stages to generate phased multi-week recovery schedules.',
    deliverable: 'Automated severity grading & timeline generator',
  },
  {
    code: 'M8',
    title: 'TFLite Edge Quantization & Offline Field Scanning',
    detail:
      'Applies float16 / INT8 TensorFlow Lite quantization to compress ResNet50 weights for low-latency mobile camera scanning in rural, low-connectivity farm environments.',
    deliverable: 'Quantized mobile edge model & field capture workflow',
  },
];

const LIMITATIONS_MAP = [
  {
    domain: 'Dataset Generalization',
    phase1Constraint:
      'PlantVillage images are captured under controlled laboratory backgrounds; real farms exhibit variable lighting, motion blur, and overlapping foliage.',
    phase2Mitigation:
      'Augment training with real-world field datasets (PlantDoc) and leaf-segmentation pre-cropping.',
  },
  {
    domain: 'Confidence Overestimation',
    phase1Constraint:
      'Standard softmax outputs can assign high probabilities to out-of-distribution or non-botanical photos.',
    phase2Mitigation:
      'Apply post-hoc Temperature Scaling and Shannon entropy thresholds to warn users on uncertain samples.',
  },
  {
    domain: 'Actionable Agronomy',
    phase1Constraint:
      'Phase 1 stops at visual classification and Grad-CAM lesion localization without generating treatment prescriptions.',
    phase2Mitigation:
      'Integrate Mistral-7B-Instruct via Hugging Face Inference API with structured pathologist prompts and safety post-processing.',
  },
  {
    domain: 'Deployment & Cold Start',
    phase1Constraint:
      'Full ResNet50 weights incur heavy memory footprints on free cloud containers and require active internet.',
    phase2Mitigation:
      'Export INT8 TFLite model for on-device offline inference and lightweight edge caching.',
  },
];

export const PhaseRoadmap: React.FC<PhaseRoadmapProps> = ({
  onOpenReport,
  onOpenDiagnostics,
}) => {
  return (
    <div className="max-w-[1440px] mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-6 border-b border-slate-200">
        <div>
          <p className="text-xs font-mono text-slate-500 tracking-wide mb-1.5">
            Two-Semester Academic Project Execution Plan
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
            Project Scope: 50% Phase 1 Built · 50% Phase 2 Planned
          </h1>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onOpenDiagnostics}
            className="px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap"
          >
            Test Phase 1 Lab
          </button>
          <button
            type="button"
            onClick={onOpenReport}
            className="px-3.5 py-2 text-xs font-medium text-white bg-teal-700 rounded-md hover:bg-teal-800 transition-colors whitespace-nowrap"
          >
            Open Semester 1 Report
          </button>
        </div>
      </div>

      {/* End-to-End Pipeline Handoff Schematic */}
      <section className="bg-white border border-slate-200 rounded-lg p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-5">
          <h2 className="text-base font-semibold text-slate-900">
            End-to-End System Pipeline &amp; Semester Boundary
          </h2>
          <span className="text-xs font-mono text-slate-500">
            Current Completion: 50% (4 of 8 Modules)
          </span>
        </div>

        {/* Progress Bar */}
        <div className="h-2.5 w-full bg-slate-100 rounded-xs overflow-hidden mb-6 flex">
          <div className="w-1/2 bg-teal-700 h-full" />
          <div className="w-1/2 bg-amber-200/70 h-full" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-6 gap-3 items-stretch">
          {[
            { step: '01', name: 'Leaf Upload & 224×224 Norm', phase: 'Semester 1', done: true },
            { step: '02', name: 'ResNet50 38-Class Inference', phase: 'Semester 1', done: true },
            { step: '03', name: 'Grad-CAM Lesion Overlay', phase: 'Semester 1', done: true },
            { step: '04', name: 'LLM Prompt Construction', phase: 'Semester 2', done: false },
            { step: '05', name: 'Mistral-7B Care Guidance', phase: 'Semester 2', done: false },
            { step: '06', name: 'Calibration & TFLite Edge', phase: 'Semester 2', done: false },
          ].map((node, idx) => (
            <div
              key={node.step}
              className={`relative p-4 rounded-md border flex flex-col justify-between ${
                node.done
                  ? 'border-teal-700/40 bg-teal-50/30'
                  : 'border-slate-200 bg-slate-50'
              }`}
            >
              <div>
                <div className="flex items-center justify-between text-[11px] font-mono mb-1.5">
                  <span className="text-slate-400">Step {node.step}</span>
                  <span
                    className={
                      node.done
                        ? 'text-teal-800 font-medium'
                        : 'text-amber-700 font-medium'
                    }
                  >
                    {node.done ? '● Built' : '○ Planned'}
                  </span>
                </div>
                <p className="text-xs font-semibold text-slate-900 leading-snug">
                  {node.name}
                </p>
              </div>
              <p className="text-[11px] text-slate-500 mt-3">{node.phase}</p>
              {idx < 5 && (
                <ArrowRight className="hidden md:block w-3.5 h-3.5 text-slate-400 absolute -right-2.5 top-1/2 -translate-y-1/2 z-10 bg-white rounded-full" />
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Side-by-Side Semester 1 vs Semester 2 Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Semester 1 Column */}
        <section className="bg-white border border-slate-200 rounded-lg p-6">
          <div className="flex items-center justify-between pb-4 border-b border-slate-200 mb-5">
            <div>
              <p className="text-xs font-mono text-teal-700 font-medium">
                Phase 1 · Current Semester (50%)
              </p>
              <h2 className="text-lg font-semibold text-slate-900 mt-0.5">
                Completed Core Vision &amp; Explainability
              </h2>
            </div>
            <span className="text-xs font-mono text-emerald-700 font-medium">
              ● Delivered
            </span>
          </div>

          <div className="space-y-5">
            {PHASE_1_MODULES.map((mod) => (
              <div
                key={mod.code}
                className="pb-5 border-b border-slate-100 last:border-b-0 last:pb-0"
              >
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded bg-teal-700 text-white text-xs font-mono flex items-center justify-center shrink-0 mt-0.5">
                    <Check className="w-3.5 h-3.5" />
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">
                      {mod.code}. {mod.title}
                    </h3>
                    <p className="text-xs text-slate-600 leading-relaxed mt-1">
                      {mod.detail}
                    </p>
                    <p className="text-xs font-mono text-teal-800 mt-2">
                      Output: {mod.deliverable}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Semester 2 Column */}
        <section className="bg-white border border-slate-200 rounded-lg p-6">
          <div className="flex items-center justify-between pb-4 border-b border-slate-200 mb-5">
            <div>
              <p className="text-xs font-mono text-amber-700 font-medium">
                Phase 2 · Next Semester (50%)
              </p>
              <h2 className="text-lg font-semibold text-slate-900 mt-0.5">
                Planned LLM Guidance &amp; Edge Deployment
              </h2>
            </div>
            <span className="text-xs font-mono text-amber-700 font-medium">
              ○ Scheduled
            </span>
          </div>

          <div className="space-y-5">
            {PHASE_2_MODULES.map((mod) => (
              <div
                key={mod.code}
                className="pb-5 border-b border-slate-100 last:border-b-0 last:pb-0"
              >
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded bg-slate-100 border border-slate-300 text-slate-600 text-xs font-mono flex items-center justify-center shrink-0 mt-0.5">
                    <Clock className="w-3.5 h-3.5" />
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">
                      {mod.code}. {mod.title}
                    </h3>
                    <p className="text-xs text-slate-600 leading-relaxed mt-1">
                      {mod.detail}
                    </p>
                    <p className="text-xs font-mono text-slate-500 mt-2">
                      Target: {mod.deliverable}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* Limitations & Next-Semester Engineering Resolution Matrix */}
      <section className="bg-white border border-slate-200 rounded-lg p-6">
        <h2 className="text-base font-semibold text-slate-900 mb-1">
          Phase 1 Limitations &amp; Semester 2 Mitigation Plan
        </h2>
        <p className="text-xs text-slate-500 mb-4">
          Key engineering constraints observed in Semester 1 that directly motivate the Semester 2 architecture.
        </p>

        <div className="overflow-x-auto border border-slate-200 rounded-md">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-medium">
                <th className="py-2.5 px-4 w-48">Engineering Area</th>
                <th className="py-2.5 px-4">Phase 1 Observed Limitation</th>
                <th className="py-2.5 px-4">Phase 2 Planned Resolution</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {LIMITATIONS_MAP.map((row) => (
                <tr key={row.domain} className="align-top">
                  <td className="py-3 px-4 font-semibold text-slate-900">
                    {row.domain}
                  </td>
                  <td className="py-3 px-4 text-slate-600 leading-relaxed">
                    {row.phase1Constraint}
                  </td>
                  <td className="py-3 px-4 text-slate-800 leading-relaxed">
                    {row.phase2Mitigation}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
