export interface LesionRegion {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  intensity: number;
  label: string;
}

export interface TopPrediction {
  classId: string;
  probability: number;
}

export interface LeafAnalysisResult {
  predictedClass: string;
  readablePrediction: string;
  crop: string;
  conditionName: string;
  pathogen: string;
  isHealthy: boolean;
  confidence: number;
  topPredictions: TopPrediction[];
  lesionRegions: LesionRegion[];
  visualObservations: string[];
  leafCoverageEstimate: number;
  analyzedAt: string;
  imageWidth: number;
  imageHeight: number;
}

export interface ReportMetadata {
  studentName: string;
  rollNumber: string;
  facultyGuide: string;
  department: string;
  institution: string;
  academicTerm: string;
}

export interface SampleSpecimen {
  id: string;
  label: string;
  crop: string;
  expectedClass: string;
  imageUrl: string;
  specimenCode: string;
  notes: string;
}
