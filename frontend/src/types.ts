export type OrganType = 'skin' | 'liver' | 'brain';

export interface SampleCase {
  id: string;
  organ: OrganType;
  title: string;
  subType: string;
  description: string;
  imageUrl: string;
  expectedMaskUrl?: string;
  resolution: string;
  provenance: string;
}

export interface OrganConfig {
  id: OrganType;
  name: string;
  datasetName: string;
  shortLabel: string;
  isAvailable: boolean;
  statusText: string;
  inputHint: string;
  modelParams: string;
  paramCount: number;
  flops: string;
  modalities: string;
  resolution: string;
  targetLesions: string;
  description: string;
  sampleCases: SampleCase[];
}

export interface PredictionMetrics {
  dice: number; // e.g. 0.9384
  confidence: number; // e.g. 0.972
  inference_time_ms: number; // e.g. 15.4
  iou: number; // Jaccard index
  sensitivity: number; // Recall
  specificity: number;
  model_size: string; // "27,312 parameters (0.11 MB)"
  parameter_count: number;
  flops_gflops: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  architecture: string;
  dataset: string;
  checkpoint: string;
  input_resolution: string;
}

export interface PredictionResult {
  id: string;
  organ_type: OrganType;
  status: 'production_ready' | 'experimental_checkpoint' | 'demo';
  original_image_url: string;
  mask_base64: string;
  metrics: PredictionMetrics;
  model_info: ModelInfo;
  timestamp: string;
  fileName?: string;
  fileSize?: string;
}

export type OverlayColor = 'red' | 'teal' | 'emerald' | 'amber';
