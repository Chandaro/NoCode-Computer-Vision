import axios from 'axios'

const api = axios.create({ baseURL: '/api' })
export default api

export interface Project { id: number; name: string; description: string; classes: string[] }

export interface ImageItem {
  id: number; filename: string; original_name: string; annotated: boolean
  width: number; height: number; color_space: string; is_corrupt: boolean; file_size: number
}

export interface AnnData {
  id?: number
  class_id: number
  shape_type: 'bbox' | 'polygon' | 'point'
  x_center: number
  y_center: number
  width: number
  height: number
  points: [number, number][]
}

// Legacy alias kept for import compat
export type BBox = AnnData

export interface TrainingRun {
  id: number; project_id: number; status: string; epochs: number; imgsz: number
  batch: number; model_base: string; model_path: string; results: Record<string, number>
  created_at: string; run_dir: string; onnx_path: string
}

export interface ClassificationRun {
  id: number; project_id: number; status: string; epochs: number; imgsz: number
  batch: number; base_model: string; lr: number; freeze_backbone: boolean
  model_path: string; results: Record<string, unknown>; created_at: string
}

export interface AnalyticsData {
  total_images: number
  annotated_images: number
  total_annotations: number
  corrupt_images: number
  class_distribution: Record<string, number>
  shape_breakdown: { bbox: number; polygon: number; point: number }
  ann_histogram: Record<string, number>
  size_samples: { w: number; h: number }[]
  aspect_buckets: Record<string, number>
  color_space_counts: Record<string, number>
  channel_stats: {
    mean: { R: number; G: number; B: number }
    std:  { R: number; G: number; B: number }
  } | null
}

export interface Detection {
  x: number; y: number; w: number; h: number
  conf: number; class_id: number; class_name: string
}

export interface InferResult {
  detections: Detection[]
  image_w: number; image_h: number; count: number
  meta?: {
    sliced: boolean; tiles: number; slice_size: number; slice_overlap: number
    tta: boolean; agnostic_nms: boolean; imgsz: number; merge: string
    raw_boxes: number; time_ms: number
  }
}

export interface ClsPrediction {
  class_id: number; class_name: string; probability: number
}

export interface ClsInferResult {
  predictions: ClsPrediction[]
  top1: ClsPrediction | null
  top5: ClsPrediction[]
}

export interface ExternalModel {
  id: number
  name: string
  model_path: string
  created_at: string
}

export interface EvalData {
  run_id: number
  model_base: string
  epochs: number
  available_plots: string[]
  csv_metrics: Record<string, number>
  overall: Record<string, number>
  per_class: Record<string, { ap50: number; precision: number; recall: number }>
  classes: string[]
}

// ── Find Anything (YOLOE-26 open-vocabulary) ─────────────────────────────────
export interface FindDetection {
  x: number; y: number; w: number; h: number   // normalised 0–1, top-left origin
  conf: number; class_id: number; class_name: string
  mask: [number, number][]                     // normalised polygon, empty if none
}

export interface FindStatus {
  models: string[]; default_model: string; downloaded: string[]
  text_encoder_ready: boolean; loaded: string[]
}
