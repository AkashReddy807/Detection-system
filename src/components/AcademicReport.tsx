import React, { useState } from 'react';
import { Download, Edit3, Printer } from 'lucide-react';
import { MODEL_BENCHMARKS, PLANT_VILLAGE_CLASSES } from '../data/plantVillage';
import type { LeafAnalysisResult, ReportMetadata } from '../types/diagnostics';

interface AcademicReportProps {
  reportMeta: ReportMetadata;
  onUpdateMeta: (next: ReportMetadata) => void;
  activeImageSrc: string | null;
  activeSpecimenLabel: string;
  gradCamSnapshotUrl: string | null;
  analysis: LeafAnalysisResult | null;
}

const TOC_ITEMS = [
  { id: 'sec-abstract', label: '1. Abstract & Problem Statement' },
  { id: 'sec-scope', label: '2. Two-Semester Scope Split' },
  { id: 'sec-dataset', label: '3. Dataset & Preprocessing' },
  { id: 'sec-architecture', label: '4. ResNet50 Transfer Learning' },
  { id: 'sec-gradcam', label: '5. Grad-CAM Explainability' },
  { id: 'sec-results', label: '6. Experimental Results' },
  { id: 'sec-phase2', label: '7. Semester 2 Roadmap' },
  { id: 'sec-references', label: '8. References' },
  { id: 'sec-appendix', label: 'Appendix A. Live Case Study' },
];

export const AcademicReport: React.FC<AcademicReportProps> = ({
  reportMeta,
  onUpdateMeta,
  activeImageSrc,
  activeSpecimenLabel,
  gradCamSnapshotUrl,
  analysis,
}) => {
  const [editingMeta, setEditingMeta] = useState(false);

  const totalImages = PLANT_VILLAGE_CLASSES.reduce((acc, c) => acc + c.datasetImages, 0);

  const buildMarkdownReport = (): string => {
    const appendixBlock = analysis
      ? `\n## Appendix A: Live Phase 1 Diagnostic Case Study\n\n- **Specimen**: ${activeSpecimenLabel}\n- **Predicted PlantVillage Class**: \`${analysis.predictedClass}\`\n- **Readable Classification**: ${analysis.readablePrediction}\n- **Causal Agent**: ${analysis.pathogen}\n- **Confidence**: ${(analysis.confidence * 100).toFixed(2)}%\n- **Estimated Affected Leaf Surface**: ${analysis.leafCoverageEstimate}%\n- **Localized Grad-CAM Focal Regions**: ${analysis.lesionRegions.length}\n- **Observed Visual Features**:\n${analysis.visualObservations.map((o) => `  - ${o}`).join('\n')}\n`
      : `\n## Appendix A: Live Phase 1 Diagnostic Case Study\n\nRun a leaf specimen in the Diagnostics Console to attach a live classification and Grad-CAM case study.\n`;

    return `# Automated Plant Disease Detection Using Deep Learning (ResNet50) and Grad-CAM Explainability
## Semester 1 (Phase 1 — 50% Completion) Technical Project Report

- **Student Name**: ${reportMeta.studentName}
- **Roll / Registration ID**: ${reportMeta.rollNumber}
- **Faculty Guide**: ${reportMeta.facultyGuide}
- **Department**: ${reportMeta.department}
- **Institution**: ${reportMeta.institution}
- **Academic Term**: ${reportMeta.academicTerm}
- **Project Milestone**: Phase 1 Completed (50%) · Phase 2 Scheduled Next Semester (50%)

---

## 1. Abstract & Problem Statement

Agricultural crop diseases cause substantial global yield losses every year. Traditional manual scouting depends on trained plant pathologists, is time-consuming, prone to subjective misdiagnosis, and cannot scale across large farms.

This two-semester project designs and implements an automated, explainable plant leaf disease detection and agronomic guidance platform. During **Semester 1 (Phase 1 — 50% scope)**, we developed and benchmarked the core computer vision and explainability pipeline: preprocessing RGB leaf imagery to 224×224 tensors, classifying 38 crop-disease categories from the PlantVillage dataset (~${totalImages.toLocaleString()} images) using a two-stage fine-tuned **ResNet50** transfer learning architecture (94.0% test accuracy), and localizing symptomatic leaf regions using **Grad-CAM** on the final convolutional block (\`conv5_block3_out\`). **Semester 2 (Phase 2 — remaining 50%)** will integrate an LLM-based plant care guidance module (\`Mistral-7B-Instruct-v0.2\` via Hugging Face Inference API), post-hoc confidence calibration (Temperature Scaling), severity scoring, and TensorFlow Lite (TFLite) mobile edge quantization.

---

## 2. Project Objectives & Two-Semester Work Breakdown

| Phase | Semester | Scope Share | Core Deliverables | Status |
| :--- | :--- | :--- | :--- | :--- |
| Phase 1 | Current Semester | 50% | PlantVillage 224×224 preprocessing, Baseline CNN vs ResNet50 Frozen & Fine-Tuned training (94% test accuracy), Grad-CAM \`conv5_block3_out\` heatmap overlay, Diagnostic Web Console, Semester 1 Report | Completed |
| Phase 2 | Next Semester | 50% | LLM-based plant health guidance (\`Mistral-7B-Instruct-v0.2\`), Temperature Scaling calibration & OOD uncertainty detection, Lesion severity scoring, TFLite quantization for offline mobile scanning | Planned |

---

## 3. Dataset Characterization & Preprocessing Pipeline

- **Source**: PlantVillage Dataset (Kaggle)
- **Scale**: ~54,000 RGB leaf photographs (${totalImages.toLocaleString()} indexed across 38 classes: 14 crop species, 26 disease classes, 12 healthy controls)
- **Split Strategy**: Stratified Train / Validation / Test = **70% / 15% / 15%**
- **Input Standardization**: Spatial resize to **224×224×3** followed by ResNet50 ImageNet channel preprocessing (\`preprocess_input\`).
- **Data Augmentation**: Random horizontal/vertical flips, controlled rotation, zoom, and brightness/contrast jitter to mitigate class imbalance across rare diseases (e.g., *Potato___healthy*, *Apple___Cedar_apple_rust*).

---

## 4. Methodology: ResNet50 Transfer Learning Architecture

\`\`\`python
inputs = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False

x = GlobalAveragePooling2D(name="gap")(base_model.output)
x = Dense(256, activation='relu', name="head_dense")(x)
x = Dropout(0.4, name="head_dropout")(x)
outputs = Dense(38, activation='softmax', name="predictions")(x)
\`\`\`

### Two-Stage Training Configuration

| Training Stage | Trainable Layers | Learning Rate | Optimizer & Loss | Regularization & Callbacks | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Stage 1 (Feature Extraction) | Frozen ResNet50 base; classification head only | \`1e-4\` | Adam, SparseCategoricalCrossentropy | Dropout(0.4), Augmentation, EarlyStopping | Removed baseline overfitting; 85% test accuracy |
| Stage 2 (Fine-Tuning) | Last 30 convolutional layers unfrozen | \`1e-5\` | Adam, SparseCategoricalCrossentropy | ReduceLROnPlateau, EarlyStopping | +9% test accuracy gain (94% validation & test) |

---

## 5. Explainable AI via Grad-CAM

To establish clinical trust and verify that the network attends to genuine pathological lesions rather than background artifacts, Gradient-weighted Class Activation Mapping (Grad-CAM) is applied to the final spatial convolutional feature map \`conv5_block3_out\` (shape \`7×7×2048\`):

1. Compute the gradient of the predicted class logit $y^c$ with respect to feature map activations $A^k$: $\\frac{\\partial y^c}{\\partial A^k}$.
2. Global-average-pool gradients over width and height to obtain neuron importance weights:
   $$\\alpha_k^c = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial y^c}{\\partial A_{i,j}^k}$$
3. Compute the weighted linear combination followed by ReLU to retain strictly positive class-discriminative regions:
   $$L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$$
4. Min-max normalize to $[0, 1]$, bilinearly resize to the original image dimensions, colorize with OpenCV \`COLORMAP_JET\`, and alpha-blend (0.6 original / 0.4 heatmap) onto the leaf specimen.

---

## 6. Phase 1 Experimental Results & Evaluation

| Model Variant | Training Accuracy | Validation Accuracy | Test Accuracy | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| Baseline Custom CNN | 83% | 77% | 79% | 6% validation gap; weak fine-grained lesion discrimination |
| ResNet50 (Frozen Base) | 70% | 89% | 85% | Eliminated overfitting; strong transfer representations |
| ResNet50 (Fine-Tuned, Last 30 Layers) | 83% | 94% | 94% | Optimal generalization; sharp Grad-CAM alignment with leaf lesions |

### Key Observations
- **Strong Generalization**: Validation and test accuracy both converge at **94%**, confirming minimal variance on held-out specimens.
- **Visual Faithfulness**: Grad-CAM activation peaks align cleanly with concentric Early Blight rings, Apple Scab blotches, and Black Rot necrotic margins while remaining suppressed on healthy chlorophyll tissue.
- **CPU Inference Readiness**: Real-time single-leaf inference executes reliably without requiring dedicated GPU acceleration.

---

## 7. Semester 2 Implementation Roadmap (Remaining 50%)

1. **LLM-Based Plant Health Guidance Engine**: Integrate \`mistralai/Mistral-7B-Instruct-v0.2\` via the Hugging Face Inference API. The Phase 1 predicted disease class and crop context will populate structured plant-pathologist prompts to generate disease-specific treatments, preventive practices, and safety-filtered care plans.
   - *Target Pipeline*: \`Leaf Image -> ResNet50 Prediction -> Disease Class -> LLM Prompt Construction -> Mistral-7B API -> Care Guidance\`
2. **Confidence Calibration & OOD Guardrails**: Apply Temperature Scaling and predictive entropy thresholds to flag blurry, low-light, or non-leaf images instead of returning overconfident predictions.
3. **Severity Scoring & Treatment Roadmaps**: Derive quantitative percentage infection grades from Grad-CAM activation masks.
4. **Edge Mobile Quantization (TFLite)**: Quantize fine-tuned ResNet50 weights for low-RAM cloud cold starts and offline rural camera scanning.

---

## 8. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778.
2. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618–626.
3. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv preprint arXiv:1511.08060 (PlantVillage Dataset).
4. Jiang, A. Q., et al. (2023). *Mistral 7B*. arXiv preprint arXiv:2310.06825.
${appendixBlock}`;
  };

  const handleDownloadMarkdown = () => {
    const content = buildMarkdownReport();
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'Phase1_Plant_Disease_Detection_Report.md';
    link.click();
    URL.revokeObjectURL(url);
  };

  const handlePrintPdf = () => {
    window.print();
  };

  return (
    <div className="max-w-[1440px] mx-auto px-6 py-8">
      {/* Top Action Bar (Hidden on Print) */}
      <div className="no-print flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-6 border-b border-slate-200 mb-8">
        <div>
          <p className="text-xs font-mono text-slate-500 tracking-wide mb-1.5">
            Academic Documentation · Submission-Ready Manuscript
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
            Semester 1 Project Report (Phase 1 — 50% Deliverable)
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={() => setEditingMeta((v) => !v)}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap"
          >
            <Edit3 className="w-3.5 h-3.5" />
            {editingMeta ? 'Hide Cover Details' : 'Edit Cover Details'}
          </button>
          <button
            type="button"
            onClick={handleDownloadMarkdown}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5" />
            Download Markdown (.md)
          </button>
          <button
            type="button"
            onClick={handlePrintPdf}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-white bg-teal-700 rounded-md hover:bg-teal-800 transition-colors whitespace-nowrap"
          >
            <Printer className="w-3.5 h-3.5" />
            Print / Save PDF
          </button>
        </div>
      </div>

      {/* Collapsible Cover Personalization Form (Hidden on Print) */}
      {editingMeta && (
        <section className="no-print bg-white border border-slate-200 rounded-lg p-5 mb-8">
          <h2 className="text-sm font-semibold text-slate-900 mb-3">
            Personalize Report Cover Sheet
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {(
              [
                { key: 'studentName', label: 'Student Name' },
                { key: 'rollNumber', label: 'Roll / Registration ID' },
                { key: 'facultyGuide', label: 'Faculty Guide / Supervisor' },
                { key: 'department', label: 'Department' },
                { key: 'institution', label: 'University / College' },
                { key: 'academicTerm', label: 'Academic Term' },
              ] as const
            ).map((field) => (
              <div key={field.key}>
                <label className="block text-xs font-medium text-slate-600 mb-1">
                  {field.label}
                </label>
                <input
                  type="text"
                  value={reportMeta[field.key]}
                  onChange={(e) =>
                    onUpdateMeta({ ...reportMeta, [field.key]: e.target.value })
                  }
                  className="w-full px-3 py-1.5 text-sm bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600"
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Layout: Sticky TOC + Formal Manuscript */}
      <div className="grid grid-cols-1 lg:grid-cols-[240px_minmax(0,1fr)] gap-8 items-start">
        {/* Sticky Table of Contents (Hidden on Print) */}
        <aside className="no-print hidden lg:block sticky top-24 bg-white border border-slate-200 rounded-lg p-4">
          <p className="text-xs font-mono text-slate-500 mb-3">Contents</p>
          <nav className="space-y-1.5">
            {TOC_ITEMS.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className="block text-xs text-slate-600 hover:text-teal-800 py-1 transition-colors"
              >
                {item.label}
              </a>
            ))}
          </nav>
        </aside>

        {/* Printable Academic Report Paper */}
        <article className="print-report-container bg-white border border-slate-200 rounded-lg px-8 sm:px-14 py-12 max-w-4xl">
          {/* Cover / Manuscript Header */}
          <header className="pb-8 border-b border-slate-300 mb-10">
            <p className="text-xs font-mono text-teal-800 tracking-wide mb-3">
              Minor / Major Project Phase 1 Progress Report · 50% Milestone Submission
            </p>
            <h1
              className="text-2xl sm:text-4xl font-bold text-slate-950 leading-tight mb-4"
              style={{ fontFamily: 'var(--font-serif)' }}
            >
              Automated Plant Disease Detection Using Deep Learning (ResNet50) and Grad-CAM Explainability
            </h1>
            <p className="text-sm text-slate-600 leading-relaxed max-w-2xl">
              Phase 1 Implementation Report covering dataset preprocessing, two-stage ResNet50 transfer learning across 38 PlantVillage classes, Grad-CAM spatial lesion visualization, and the architectural blueprint for Semester 2 LLM-based plant care guidance.
            </p>

            {/* Academic Cover Metadata Grid */}
            <dl className="mt-8 pt-6 border-t border-slate-200 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-4 text-xs">
              <div>
                <dt className="text-slate-500">Submitted By</dt>
                <dd className="font-semibold text-slate-900 mt-0.5">
                  {reportMeta.studentName}
                </dd>
              </div>
              <div>
                <dt className="text-slate-500">Roll / ID Number</dt>
                <dd className="font-mono text-slate-900 mt-0.5">
                  {reportMeta.rollNumber}
                </dd>
              </div>
              <div>
                <dt className="text-slate-500">Project Guide</dt>
                <dd className="font-semibold text-slate-900 mt-0.5">
                  {reportMeta.facultyGuide}
                </dd>
              </div>
              <div>
                <dt className="text-slate-500">Department</dt>
                <dd className="text-slate-800 mt-0.5">{reportMeta.department}</dd>
              </div>
              <div>
                <dt className="text-slate-500">Institution</dt>
                <dd className="text-slate-800 mt-0.5">{reportMeta.institution}</dd>
              </div>
              <div>
                <dt className="text-slate-500">Academic Term</dt>
                <dd className="font-mono text-slate-800 mt-0.5">
                  {reportMeta.academicTerm}
                </dd>
              </div>
            </dl>
          </header>

          {/* Report Prose Sections */}
          <div className="space-y-10 text-[15px] leading-7 text-slate-800">
            {/* Section 1 */}
            <section id="sec-abstract" className="scroll-mt-24">
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                1. Abstract &amp; Problem Statement
              </h2>
              <p className="mb-3">
                Crop leaf diseases inflict major agricultural yield and economic losses worldwide. Conventional field diagnosis relies heavily on manual inspection by trained plant pathologists—a workflow that is slow, costly, susceptible to human fatigue, and impractical across large farming acreage.
              </p>
              <p>
                This project addresses these challenges through a two-semester deep learning and explainable AI software system. In <strong>Semester 1 (Phase 1 — 50% completion)</strong>, we engineered the end-to-end visual classification and interpretability core: standardizing leaf photographs into <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">224×224×3</code> tensors, fine-tuning an ImageNet-pretrained <strong>ResNet50</strong> architecture across <strong>38 PlantVillage classes</strong> to reach <strong>94.0% test accuracy</strong>, and generating real-time <strong>Grad-CAM</strong> heatmaps from <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">conv5_block3_out</code> to highlight exact symptomatic leaf tissue. <strong>Semester 2 (Phase 2 — remaining 50%)</strong> will extend this diagnostic core with contextual LLM-based plant care guidance (<code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">Mistral-7B-Instruct-v0.2</code>), confidence calibration via Temperature Scaling, severity scoring, and quantized TensorFlow Lite mobile deployment.
              </p>
            </section>

            {/* Section 2 */}
            <section id="sec-scope" className="scroll-mt-24 print-break-avoid">
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                2. Project Objectives &amp; Two-Semester Scope Split
              </h2>
              <p className="mb-4">
                To maintain rigorous validation discipline across the full academic year, development is partitioned evenly between computer vision explainability in Semester 1 and generative agronomic guidance plus edge optimization in Semester 2.
              </p>

              <div className="overflow-x-auto border border-slate-200 rounded-md">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 text-slate-700 font-semibold">
                      <th className="py-2.5 px-3">Phase</th>
                      <th className="py-2.5 px-3">Share</th>
                      <th className="py-2.5 px-3">Implemented / Planned Modules</th>
                      <th className="py-2.5 px-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    <tr className="align-top">
                      <td className="py-3 px-3 font-semibold text-slate-900 whitespace-nowrap">
                        Phase 1 (Semester 1)
                      </td>
                      <td className="py-3 px-3 font-mono tabular-nums">50%</td>
                      <td className="py-3 px-3 leading-relaxed">
                        PlantVillage dataset stratification &amp; 224×224 preprocessing; Baseline CNN, Frozen ResNet50, and Fine-Tuned ResNet50 evaluation; Grad-CAM spatial lesion highlighting; Interactive Web Diagnostic Console; Phase 1 Technical Report.
                      </td>
                      <td className="py-3 px-3 font-medium text-emerald-700 whitespace-nowrap">
                        ● Completed
                      </td>
                    </tr>
                    <tr className="align-top">
                      <td className="py-3 px-3 font-semibold text-slate-900 whitespace-nowrap">
                        Phase 2 (Semester 2)
                      </td>
                      <td className="py-3 px-3 font-mono tabular-nums">50%</td>
                      <td className="py-3 px-3 leading-relaxed">
                        Hugging Face LLM inference pipeline (Mistral-7B-Instruct-v0.2) for dynamic treatment and prevention guidance; Temperature Scaling confidence calibration &amp; OOD rejection; Lesion severity grading; TFLite INT8 quantization for offline mobile scanning.
                      </td>
                      <td className="py-3 px-3 font-medium text-amber-700 whitespace-nowrap">
                        ○ Next Semester
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* Section 3 */}
            <section id="sec-dataset" className="scroll-mt-24 print-break-avoid">
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                3. Dataset Characterization &amp; Preprocessing Pipeline
              </h2>
              <p className="mb-3">
                The classifier is trained and evaluated on the open-access <strong>PlantVillage Dataset</strong> (Kaggle), comprising approximately <strong>54,000 RGB leaf images</strong> ({totalImages.toLocaleString()} curated images across <strong>38 distinct categories</strong> covering 14 crops including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato).
              </p>
              <ul className="list-disc pl-6 space-y-1.5 text-sm text-slate-700">
                <li>
                  <strong>Stratified Partitioning</strong>: Divided into 70% Training, 15% Validation, and 15% Hold-out Test splits preserving minority class proportions.
                </li>
                <li>
                  <strong>Spatial &amp; Channel Preprocessing</strong>: Every input leaf image is converted to RGB, resized to <span className="font-mono">224×224</span>, batched, and normalized via Keras <span className="font-mono">resnet50.preprocess_input</span>.
                </li>
                <li>
                  <strong>Class Imbalance Mitigation</strong>: Online data augmentation applies random horizontal/vertical flips, rotations, zoom scaling, and brightness/contrast perturbations during training.
                </li>
              </ul>
            </section>

            {/* Section 4 */}
            <section id="sec-architecture" className="scroll-mt-24 print-break-avoid">
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                4. Methodology: ResNet50 Transfer Learning Architecture
              </h2>
              <p className="mb-4">
                Training a deep convolutional network from scratch on agricultural datasets frequently overfits due to inter-class visual similarity and class imbalance. We leverage <strong>ResNet50</strong> initialized with ImageNet weights (<code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">include_top=False</code>) and attach a regularized classification head:
              </p>

              <pre className="p-4 bg-slate-50 border border-slate-200 rounded-md text-xs font-mono text-slate-800 overflow-x-auto leading-relaxed mb-5">
{`inputs = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
base_model.trainable = False

x = GlobalAveragePooling2D(name="gap")(base_model.output)
x = Dense(256, activation='relu', name="head_dense")(x)
x = Dropout(0.4, name="head_dropout")(x)
outputs = Dense(38, activation='softmax', name="predictions")(x)`}
              </pre>

              <p className="mb-3">
                Optimization proceeds in two controlled stages using the <strong>Adam</strong> optimizer, <strong>SparseCategoricalCrossentropy</strong> loss, <strong>EarlyStopping</strong>, and <strong>ReduceLROnPlateau</strong>:
              </p>
              <div className="overflow-x-auto border border-slate-200 rounded-md">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 text-slate-700 font-semibold">
                      <th className="py-2.5 px-3">Stage</th>
                      <th className="py-2.5 px-3">Trainable Layers</th>
                      <th className="py-2.5 px-3 font-mono">Learning Rate</th>
                      <th className="py-2.5 px-3">Objective</th>
                      <th className="py-2.5 px-3">Observed Impact</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    <tr>
                      <td className="py-2.5 px-3 font-medium">Phase 1</td>
                      <td className="py-2.5 px-3">Frozen ResNet50 backbone; custom head only</td>
                      <td className="py-2.5 px-3 font-mono">1e-4</td>
                      <td className="py-2.5 px-3">Rapid head convergence</td>
                      <td className="py-2.5 px-3">Eliminated baseline overfitting (85% test)</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-3 font-medium">Phase 2</td>
                      <td className="py-2.5 px-3">Last 30 layers of Conv5 unfrozen</td>
                      <td className="py-2.5 px-3 font-mono">1e-5</td>
                      <td className="py-2.5 px-3">Botanical domain adaptation</td>
                      <td className="py-2.5 px-3">+9% test accuracy gain (94% val / 94% test)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* Section 5 */}
            <section id="sec-gradcam" className="scroll-mt-24 print-break-avoid">
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                5. Explainable AI via Grad-CAM
              </h2>
              <p className="mb-3">
                Black-box disease predictions offer insufficient transparency for agronomists and farmers. We integrate <strong>Gradient-weighted Class Activation Mapping (Grad-CAM)</strong> targeting the final residual convolutional output <code className="font-mono text-xs bg-slate-100 px-1.5 py-0.5 rounded">conv5_block3_out</code> (spatial resolution <span className="font-mono">7×7×2048</span>), which retains the richest semantic pathology features alongside coarse spatial geometry.
              </p>
              <div className="p-4 bg-slate-50 border border-slate-200 rounded-md font-mono text-xs text-slate-800 space-y-2 mb-3">
                <p>1. Importance Weight:  α_k^c = (1 / Z) · Σ_i Σ_j ( ∂y^c / ∂A_ij^k )</p>
                <p>2. Localization Map:   L_Grad-CAM^c = ReLU( Σ_k α_k^c · A^k )</p>
                <p>3. Overlay Synthesis:  I_out = 0.6 · I_leaf + 0.4 · COLORMAP_JET( Norm(L_Grad-CAM^c) )</p>
              </div>
              <p>
                Applying ReLU removes negative activations belonging to competing disease classes, isolating strictly the necrotic spots, fungal pustules, or chlorotic halos driving the winning softmax logit.
              </p>
            </section>

            {/* Section 6 */}
            <section id="sec-results" className="scroll-mt-24 print-break-avoid">
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                6. Phase 1 Experimental Results &amp; Evaluation
              </h2>
              <p className="mb-4">
                Table 3 summarizes empirical accuracy across the three progressively refined architectures evaluated on the stratified PlantVillage splits.
              </p>

              <div className="overflow-x-auto border border-slate-200 rounded-md mb-4">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 text-slate-700 font-semibold">
                      <th className="py-2.5 px-3">Model Architecture</th>
                      <th className="py-2.5 px-3 text-right font-mono">Training Acc</th>
                      <th className="py-2.5 px-3 text-right font-mono">Validation Acc</th>
                      <th className="py-2.5 px-3 text-right font-mono">Test Acc</th>
                      <th className="py-2.5 px-3 text-right font-mono">Macro F1</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
                    {MODEL_BENCHMARKS.map((row) => (
                      <tr key={row.id}>
                        <td className="py-2.5 px-3 font-sans font-medium text-slate-900">
                          {row.name}
                        </td>
                        <td className="py-2.5 px-3 text-right">{row.trainAcc.toFixed(1)}%</td>
                        <td className="py-2.5 px-3 text-right">{row.valAcc.toFixed(1)}%</td>
                        <td className="py-2.5 px-3 text-right font-semibold text-teal-800">
                          {row.testAcc.toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right">{row.macroF1.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <ul className="list-disc pl-6 space-y-1.5 text-sm text-slate-700">
                <li>
                  <strong>Superior Generalization</strong>: Fine-tuning the final 30 layers raised held-out test accuracy by <strong>+15 percentage points</strong> over the baseline CNN (79% → 94%) with zero validation-to-test degradation.
                </li>
                <li>
                  <strong>Grad-CAM Verification</strong>: Qualitative inspection confirms activation heatmaps concentrate accurately on disease lesions while staying low and diffuse across healthy leaf blades.
                </li>
                <li>
                  <strong>Real-Time Execution</strong>: Single-image preprocessing, inference, and Grad-CAM rendering complete interactively without requiring dedicated GPU hardware.
                </li>
              </ul>
            </section>

            {/* Section 7 */}
            <section id="sec-phase2" className="scroll-mt-24 print-break-avoid">
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                7. Semester 2 Implementation Roadmap (Remaining 50%)
              </h2>
              <p className="mb-3">
                With the visual detection and Grad-CAM explainability core established in Semester 1, the upcoming semester will complete the remaining half of the system:
              </p>
              <ol className="list-decimal pl-6 space-y-2 text-sm text-slate-700">
                <li>
                  <strong>LLM-Based Plant Health Recommendation Engine</strong>: Connect <span className="font-mono">mistralai/Mistral-7B-Instruct-v0.2</span> via the Hugging Face Inference API. The Phase 1 predicted class and lesion coverage will populate structured plant-pathologist prompts to deliver actionable treatment protocols, organic/chemical management options, and preventive hygiene measures.
                </li>
                <li>
                  <strong>Confidence Calibration &amp; Uncertainty Detection</strong>: Implement Temperature Scaling and Shannon entropy thresholds to detect out-of-distribution photographs (blurry backgrounds, non-leaf objects, or unsupported crops).
                </li>
                <li>
                  <strong>Disease Severity Scoring</strong>: Convert Grad-CAM affected surface percentages into standardized clinical severity grades (Mild, Moderate, Severe) linked to phased treatment timelines.
                </li>
                <li>
                  <strong>Mobile Edge Quantization (TFLite)</strong>: Export quantized weights for low-memory cloud execution and offline camera scanning in field settings.
                </li>
              </ol>
            </section>

            {/* Section 8 */}
            <section id="sec-references" className="scroll-mt-24 print-break-avoid">
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                8. References
              </h2>
              <ol className="list-decimal pl-6 space-y-2 text-xs text-slate-700 leading-relaxed">
                <li>
                  He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep Residual Learning for Image Recognition. <em>IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</em>, pp. 770–778.
                </li>
                <li>
                  Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &amp; Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. <em>IEEE International Conference on Computer Vision (ICCV)</em>, pp. 618–626.
                </li>
                <li>
                  Hughes, D. P., &amp; Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics (PlantVillage Dataset). <em>arXiv:1511.08060</em>.
                </li>
                <li>
                  Jiang, A. Q., Sablayrolles, A., Mensch, A., et al. (2023). Mistral 7B. <em>arXiv:2310.06825</em>.
                </li>
              </ol>
            </section>

            {/* Dynamic Appendix A: Live Case Study */}
            <section
              id="sec-appendix"
              className="scroll-mt-24 pt-8 border-t border-slate-300 print-break-avoid"
            >
              <h2
                className="text-xl font-bold text-slate-950 mb-3"
                style={{ fontFamily: 'var(--font-serif)' }}
              >
                Appendix A. Live Diagnostic &amp; Grad-CAM Case Study
              </h2>

              {analysis && activeImageSrc ? (
                <div className="space-y-4">
                  <p className="text-sm text-slate-700">
                    The following live diagnostic sample was captured directly from the Phase 1 Diagnostics Console to demonstrate classification output alongside Grad-CAM spatial lesion localization.
                  </p>

                  <div className="grid grid-cols-2 gap-4 max-w-xl">
                    <div>
                      <div className="aspect-square rounded border border-slate-200 overflow-hidden bg-slate-50">
                        <img
                          src={activeImageSrc}
                          alt={`Original ${activeSpecimenLabel}`}
                          referrerPolicy="no-referrer"
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <p className="text-[11px] font-mono text-slate-500 mt-1.5">
                        Figure A.1 — Input Leaf Specimen ({activeSpecimenLabel})
                      </p>
                    </div>
                    <div>
                      <div className="aspect-square rounded border border-slate-200 overflow-hidden bg-slate-50">
                        {gradCamSnapshotUrl ? (
                          <img
                            src={gradCamSnapshotUrl}
                            alt="Grad-CAM lesion overlay"
                            referrerPolicy="no-referrer"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-xs text-slate-400">
                            Grad-CAM snapshot ready on Diagnostics tab
                          </div>
                        )}
                      </div>
                      <p className="text-[11px] font-mono text-slate-500 mt-1.5">
                        Figure A.2 — Grad-CAM Overlay (conv5_block3_out)
                      </p>
                    </div>
                  </div>

                  <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 bg-slate-50 border border-slate-200 rounded-md text-xs">
                    <div>
                      <dt className="text-slate-500">Predicted Class</dt>
                      <dd className="font-semibold text-slate-900 mt-0.5">
                        {analysis.crop} — {analysis.conditionName}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-slate-500">Confidence</dt>
                      <dd className="font-mono tabular-nums font-semibold text-teal-800 mt-0.5">
                        {(analysis.confidence * 100).toFixed(2)}%
                      </dd>
                    </div>
                    <div>
                      <dt className="text-slate-500">Affected Area</dt>
                      <dd className="font-mono tabular-nums text-slate-900 mt-0.5">
                        {analysis.leafCoverageEstimate}%
                      </dd>
                    </div>
                    <div>
                      <dt className="text-slate-500">Status</dt>
                      <dd className="font-medium text-slate-900 mt-0.5">
                        {analysis.isHealthy ? '● Healthy' : '▲ Diseased'}
                      </dd>
                    </div>
                  </dl>
                </div>
              ) : (
                <p className="text-sm text-slate-600">
                  Analyze any leaf image in the Diagnostics Console to automatically attach its live classification metrics and Grad-CAM heatmap overlay to this report appendix.
                </p>
              )}
            </section>
          </div>
        </article>
      </div>
    </div>
  );
};
