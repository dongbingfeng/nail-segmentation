export interface SAMHealthResponse {
  status: "healthy" | "unhealthy" | "loading";
  model_loaded: boolean;
  model_type: string;
  device: string;
  message?: string;
  gpu_available?: boolean;
  gpu_memory_allocated?: number;
  gpu_memory_cached?: number;
  error?: string;
}

export interface SAMModelInfo {
  initialized: boolean;
  model_type: string;
  device: string;
  confidence_threshold: number;
  max_points: number;
  loaded_models?: string[];
  model_count?: number;
  gpu_memory_allocated?: number;
  gpu_memory_cached?: number;
}

export interface SAMInitializeResponse {
  message: string;
  success: boolean;
} 