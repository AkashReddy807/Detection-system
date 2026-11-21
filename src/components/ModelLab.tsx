import React, { useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import {
  MODEL_BENCHMARKS,
  PLANT_VILLAGE_CLASSES,
  RESNET50_ARCHITECTURE,
  TRAINING_HISTORY,
  UNIQUE_CROPS,
} from '../data/plantVillage';

export const ModelLab: React.FC = () => {
  const [selectedLayerId, setSelectedLayerId] = useState<string>('conv5');
  const [chartMetric, setChartMetric] = useState<'accuracy' | 'loss'>('accuracy');
  const [hoveredEpoch, setHoveredEpoch] = useState<number | null>(null);
  const [cropFilter, setCropFilter] = useState<string>('All');
  const [statusFilter, setStatusFilter] = useState<'all' | 'diseased' | 'healthy'>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const selectedLayer =
    RESNET50_ARCHITECTURE.find((l) => l.id === selectedLayerId) ??
    RESNET50_ARCHITECTURE[2];

  const filteredClasses = useMemo(() => {
    return PLANT_VILLAGE_CLASSES.filter((item) => {
      if (cropFilter !== 'All' && item.crop !== cropFilter) return false;
      if (statusFilter === 'diseased' && item.isHealthy) return false;
      if (statusFilter === 'healthy' && !item.isHealthy) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return (
          item.id.toLowerCase().includes(q) ||
          item.crop.toLowerCase().includes(q) ||
          item.condition.toLowerCase().includes(q) ||
          item.pathogen.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [cropFilter, statusFilter, searchQuery]);

  // SVG Training Curve coordinates
  const svgW = 620;
  const svgH = 260;
  const padL = 48;
  const padR = 20;
  const padT = 20;
  const padB = 34;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;

  const getX = (epoch: number) => padL + ((epoch - 1) / (TRAINING_HISTORY.length - 1)) * plotW;
  const getY = (val: number) => {
    if (chartMetric === 'accuracy') {
      const minY = 30;
      const maxY = 100;
      return padT + ((maxY - val) / (maxY - minY)) * plotH;
    }
    const minY = 0;
    const maxY = 2.5;
    return padT + ((maxY - val) / (maxY - minY)) * plotH;
  };

  const trainPoints = TRAINING_HISTORY.map(
    (p) => `${getX(p.epoch).toFixed(1)},${getY(chartMetric === 'accuracy' ? p.trainAcc : p.trainLoss).toFixed(1)}`
  ).join(' ');

  const valPoints = TRAINING_HISTORY.map(
    (p) => `${getX(p.epoch).toFixed(1)},${getY(chartMetric === 'accuracy' ? p.valAcc : p.valLoss).toFixed(1)}`
  ).join(' ');

  const phase2BoundaryX = getX(8.5);
  const activePoint =
    hoveredEpoch !== null
      ? TRAINING_HISTORY.find((p) => p.epoch === hoveredEpoch) ?? null
      : null;

  const totalDatasetImages = PLANT_VILLAGE_CLASSES.reduce(
    (sum, c) => sum + c.datasetImages,
    0
  );

  return (
    <div className="max-w-[1440px] mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-6 border-b border-slate-200">
        <div>
          <p className="text-xs font-mono text-slate-500 tracking-wide mb-1.5">
            Deep Learning Pipeline · Transfer Learning &amp; Evaluation
          </p>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
            ResNet50 Architecture &amp; Benchmark Lab
          </h1>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono tabular-nums text-slate-500">
          <span>Best Test Accuracy: 94.0%</span>
          <span aria-hidden="true">·</span>
          <span>PlantVillage {totalDatasetImages.toLocaleString()} Images</span>
          <span aria-hidden="true">·</span>
          <span>Split 70 / 15 / 15</span>
        </div>
      </div>

      {/* Section 1: Interactive ResNet50 Transfer Learning Architecture */}
      <section className="bg-white border border-slate-200 rounded-lg p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-5">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              01. ResNet50 Transfer Learning Architecture
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Click any network stage to inspect tensor dimensions, parameter counts, and two-phase training configuration.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            Optimizer: Adam · Loss: SparseCategoricalCrossentropy
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6 items-start">
          {/* Layer Pipeline Schematic */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
            {RESNET50_ARCHITECTURE.map((layer, index) => {
              const isSelected = layer.id === selectedLayer.id;
              return (
                <button
                  key={layer.id}
                  type="button"
                  onClick={() => setSelectedLayerId(layer.id)}
                  className={`text-left p-4 rounded-md border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 ${
                    isSelected
                      ? 'border-teal-700 bg-teal-50/40'
                      : 'border-slate-200 bg-slate-50/50 hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 mb-1.5">
                    <span>Stage 0{index + 1}</span>
                    {layer.isGradCamTarget && (
                      <span className="text-teal-700 font-medium">Grad-CAM</span>
                    )}
                  </div>
                  <p className="text-sm font-semibold text-slate-900 leading-snug">
                    {layer.name}
                  </p>
                  <p className="text-xs font-mono text-slate-600 mt-2 truncate">
                    {layer.outputShape}
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1">
                    Params: <span className="font-mono tabular-nums">{layer.params}</span>
                  </p>
                </button>
              );
            })}
          </div>

          {/* Layer Specification Inspector */}
          <div className="bg-slate-50 border border-slate-200 rounded-md p-5 space-y-4">
            <div>
              <p className="text-xs font-mono text-teal-700 mb-1">
                Selected Layer Specification
              </p>
              <h3 className="text-base font-semibold text-slate-900">
                {selectedLayer.name}
              </h3>
              <code className="block mt-2 p-2.5 bg-white border border-slate-200 rounded text-xs font-mono text-slate-800 break-all">
                {selectedLayer.kerasLayer}
              </code>
            </div>

            <dl className="text-xs space-y-2.5 border-t border-slate-200 pt-3">
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">Output Tensor</dt>
                <dd className="font-mono text-slate-900">{selectedLayer.outputShape}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">Parameters</dt>
                <dd className="font-mono tabular-nums text-slate-900">
                  {selectedLayer.params}
                </dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">Phase-1 (Frozen)</dt>
                <dd className="font-mono text-slate-900">{selectedLayer.phase1State}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-slate-500">Phase-2 (Fine-Tune)</dt>
                <dd className="font-mono text-teal-800 font-medium">
                  {selectedLayer.phase2State}
                </dd>
              </div>
            </dl>

            <p className="text-xs text-slate-600 leading-relaxed border-t border-slate-200 pt-3">
              {selectedLayer.role}
            </p>
          </div>
        </div>
      </section>

      {/* Section 2: Empirical Benchmark Comparison & Training Curves */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Model Benchmark Table & Bars */}
        <section className="bg-white border border-slate-200 rounded-lg p-6">
          <h2 className="text-base font-semibold text-slate-900 mb-1">
            02. Progressive Model Evaluation
          </h2>
          <p className="text-xs text-slate-500 mb-5">
            Comparative performance across custom CNN baseline, frozen ResNet50, and fine-tuned ResNet50 (last 30 layers unfrozen).
          </p>

          <div className="space-y-5">
            {MODEL_BENCHMARKS.map((m) => (
              <div key={m.id} className="pb-4 border-b border-slate-100 last:border-b-0 last:pb-0">
                <div className="flex flex-wrap items-baseline justify-between gap-2 mb-2">
                  <div>
                    <p className="text-sm font-semibold text-slate-900">{m.name}</p>
                    <p className="text-xs text-slate-500">
                      {m.stage} · lr = {m.learningRate}
                    </p>
                  </div>
                  <div className="flex items-center gap-4 text-xs font-mono tabular-nums">
                    <span className="text-slate-500">Train {m.trainAcc}%</span>
                    <span className="text-slate-700">Val {m.valAcc}%</span>
                    <span className="font-semibold text-teal-800">Test {m.testAcc}%</span>
                  </div>
                </div>

                {/* Triple Split Bar */}
                <div className="grid grid-cols-3 gap-2 mt-2">
                  <div>
                    <div className="h-2 bg-slate-100 rounded-xs overflow-hidden">
                      <div
                        className="h-full bg-slate-400"
                        style={{ width: `${m.trainAcc}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="h-2 bg-slate-100 rounded-xs overflow-hidden">
                      <div
                        className="h-full bg-slate-600"
                        style={{ width: `${m.valAcc}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="h-2 bg-slate-100 rounded-xs overflow-hidden">
                      <div
                        className="h-full bg-teal-700"
                        style={{ width: `${m.testAcc}%` }}
                      />
                    </div>
                  </div>
                </div>
                <p className="text-xs text-slate-600 mt-2">{m.notes}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Two-Phase Training Progression SVG Chart */}
        <section className="bg-white border border-slate-200 rounded-lg p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                03. Two-Phase Training Convergence
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Epochs 1–8 Frozen Base (lr 1e-4) · Epochs 9–15 Fine-Tuning (lr 1e-5)
              </p>
            </div>
            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
              <button
                type="button"
                onClick={() => setChartMetric('accuracy')}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                  chartMetric === 'accuracy'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Accuracy (%)
              </button>
              <button
                type="button"
                onClick={() => setChartMetric('loss')}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                  chartMetric === 'loss'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Loss
              </button>
            </div>
          </div>

          <div className="relative">
            <svg
              viewBox={`0 0 ${svgW} ${svgH}`}
              className="w-full h-auto select-none"
              role="img"
              aria-label="ResNet50 training and validation convergence chart"
            >
              {/* Horizontal Grid Lines */}
              {(chartMetric === 'accuracy'
                ? [40, 60, 80, 100]
                : [0.5, 1.0, 1.5, 2.0, 2.5]
              ).map((tick) => {
                const y = getY(tick);
                return (
                  <g key={tick}>
                    <line
                      x1={padL}
                      y1={y}
                      x2={svgW - padR}
                      y2={y}
                      stroke="#e2e8f0"
                      strokeWidth="1"
                    />
                    <text
                      x={padL - 8}
                      y={y + 4}
                      textAnchor="end"
                      className="fill-slate-500 text-[10px] font-mono"
                    >
                      {chartMetric === 'accuracy' ? `${tick}%` : tick.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Phase 2 Unfreeze Boundary */}
              <line
                x1={phase2BoundaryX}
                y1={padT}
                x2={phase2BoundaryX}
                y2={svgH - padB}
                stroke="#0f766e"
                strokeWidth="1"
                strokeDasharray="4 3"
              />
              <text
                x={phase2BoundaryX + 6}
                y={padT + 12}
                className="fill-teal-800 text-[10px] font-mono"
              >
                Unfreeze last 30 layers
              </text>

              {/* Validation Curve (Solid Teal) */}
              <polyline
                fill="none"
                stroke="#0f766e"
                strokeWidth="2.2"
                points={valPoints}
              />

              {/* Training Curve (Dashed Slate) */}
              <polyline
                fill="none"
                stroke="#64748b"
                strokeWidth="1.8"
                strokeDasharray="5 3"
                points={trainPoints}
              />

              {/* Interactive Epoch Hover Targets */}
              {TRAINING_HISTORY.map((pt) => {
                const cx = getX(pt.epoch);
                const valY = getY(chartMetric === 'accuracy' ? pt.valAcc : pt.valLoss);
                const trainY = getY(chartMetric === 'accuracy' ? pt.trainAcc : pt.trainLoss);
                const isHov = hoveredEpoch === pt.epoch;
                return (
                  <g key={pt.epoch}>
                    {isHov && (
                      <line
                        x1={cx}
                        y1={padT}
                        x2={cx}
                        y2={svgH - padB}
                        stroke="#94a3b8"
                        strokeWidth="1"
                      />
                    )}
                    <circle
                      cx={cx}
                      cy={valY}
                      r={isHov ? 4.5 : 2.8}
                      className="fill-teal-700"
                    />
                    <circle
                      cx={cx}
                      cy={trainY}
                      r={isHov ? 4 : 2.4}
                      className="fill-slate-500"
                    />
                    <text
                      x={cx}
                      y={svgH - 10}
                      textAnchor="middle"
                      className="fill-slate-500 text-[10px] font-mono"
                    >
                      {pt.epoch}
                    </text>
                    <rect
                      x={cx - plotW / 30}
                      y={padT}
                      width={plotW / 15}
                      height={plotH}
                      fill="transparent"
                      className="cursor-crosshair"
                      onMouseEnter={() => setHoveredEpoch(pt.epoch)}
                      onMouseLeave={() => setHoveredEpoch(null)}
                    />
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Chart Legend & Active Crosshair Readout */}
          <div className="mt-3 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-5 text-slate-600">
              <span className="inline-flex items-center gap-2">
                <span className="w-4 h-0.5 bg-teal-700 inline-block" />
                Validation
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="w-4 h-0.5 border-t-2 border-dashed border-slate-500 inline-block" />
                Training
              </span>
            </div>
            <div className="font-mono tabular-nums text-slate-700">
              {activePoint
                ? `Epoch ${activePoint.epoch} (${activePoint.phase}) · Train ${
                    chartMetric === 'accuracy'
                      ? `${activePoint.trainAcc.toFixed(1)}%`
                      : activePoint.trainLoss.toFixed(2)
                  } · Val ${
                    chartMetric === 'accuracy'
                      ? `${activePoint.valAcc.toFixed(1)}%`
                      : activePoint.valLoss.toFixed(2)
                  }`
                : 'Hover over epochs to inspect values'}
            </div>
          </div>
        </section>
      </div>

      {/* Section 3: 38-Class PlantVillage Dataset Taxonomy Explorer */}
      <section className="bg-white border border-slate-200 rounded-lg p-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-5">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              04. PlantVillage 38-Class Taxonomy Catalog
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Complete output space of the fine-tuned ResNet50 softmax classifier (14 crops · 26 disease classes · 12 healthy controls).
            </p>
          </div>

          {/* Filter Bar */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search class or pathogen..."
                className="pl-8 pr-3 py-1.5 text-xs bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600 w-52"
              />
            </div>

            <select
              value={cropFilter}
              onChange={(e) => setCropFilter(e.target.value)}
              aria-label="Filter by crop species"
              className="px-3 py-1.5 text-xs bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-600"
            >
              <option value="All">All Crops (14)</option>
              {UNIQUE_CROPS.map((crop) => (
                <option key={crop} value={crop}>
                  {crop}
                </option>
              ))}
            </select>

            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-md">
              {(['all', 'diseased', 'healthy'] as const).map((st) => (
                <button
                  key={st}
                  type="button"
                  onClick={() => setStatusFilter(st)}
                  className={`px-2.5 py-1 text-xs font-medium rounded capitalize transition-colors whitespace-nowrap ${
                    statusFilter === st
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {st}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="overflow-x-auto border border-slate-200 rounded-md">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-600 font-medium">
                <th className="py-2.5 px-3 font-mono w-12">Idx</th>
                <th className="py-2.5 px-3">Crop</th>
                <th className="py-2.5 px-3">Condition</th>
                <th className="py-2.5 px-3">Causal Agent</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3 text-right font-mono">Samples</th>
                <th className="py-2.5 px-3">Key Visual Signature</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredClasses.map((item) => (
                <tr key={item.id} className="hover:bg-slate-50/80 transition-colors">
                  <td className="py-2.5 px-3 font-mono tabular-nums text-slate-400">
                    {String(item.index).padStart(2, '0')}
                  </td>
                  <td className="py-2.5 px-3 font-medium text-slate-900 whitespace-nowrap">
                    {item.crop}
                  </td>
                  <td className="py-2.5 px-3 text-slate-800 whitespace-nowrap">
                    {item.condition}
                  </td>
                  <td className="py-2.5 px-3 text-slate-600 italic">
                    {item.pathogen}
                  </td>
                  <td className="py-2.5 px-3 whitespace-nowrap">
                    <span
                      className={
                        item.isHealthy
                          ? 'text-emerald-700 font-medium'
                          : 'text-rose-700 font-medium'
                      }
                    >
                      {item.isHealthy ? '● Healthy' : '▲ Diseased'}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-700">
                    {item.datasetImages.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 text-slate-600 max-w-xs">
                    {item.visualSignature}
                  </td>
                </tr>
              ))}
              {filteredClasses.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-500">
                    No PlantVillage classes match the current filter criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-slate-500 mt-3 font-mono tabular-nums">
          Showing {filteredClasses.length} of {PLANT_VILLAGE_CLASSES.length} classes
        </p>
      </section>
    </div>
  );
};
