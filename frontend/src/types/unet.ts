/**
 * TypeScript interfaces for U-Net API communication
 * These mirror the Pydantic models from the backend
 */

export interface UNetSegmentationRequest {
  image_data: string;
  threshold?: number;
  return_confidence?: boolean;
  return_contours?: boolean;
  refine_mask?: boolean;
  return_visualizations?: boolean;
}

export interface ConfidenceScores {
  overall_mean: number;
  overall_max: number;
  overall_min: number;
  mask_mean: number;
  mask_min: number;
  mask_max: number;
  background_mean: number;
  background_max: number;
  mean_certainty: number;
}

export interface ContourData {
  points: number[][];
  area: number;
  bounding_box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface VisualizationOutputs {
  binary_mask_vis?: string;
  confidence_heatmap?: string;
  contour_mask?: string;
  combined_overlay?: string;
}

export interface UNetSegmentationResponse {
  success: boolean;
  request_id?: string;
  processing_time_ms?: number;
  
  // Core segmentation results
  mask_data?: string;
  binary_mask?: string;
  mask_area?: number;
  mask_area_ratio?: number;
  threshold_used?: number;
  model_info?: Record<string, any>;
  
  // Confidence information (optional)
  confidence_scores?: ConfidenceScores;
  confidence_map?: string;
  
  // Contour information (optional)
  contours?: ContourData[];
  largest_contour_area?: number;
  bounding_boxes?: BoundingBox[];
  
  // Visualization outputs (optional)
  visualizations?: VisualizationOutputs;
  
  // Error information
  error?: Record<string, any>;
}

export interface UNetHealthResponse {
  service_healthy: boolean;
  model_loaded: boolean;
  memory_pools_ready: boolean;
  gpu_available: boolean;
  last_health_check: string;
  model_info?: Record<string, any>;
  memory_usage?: Record<string, any>;
  capabilities: string[];
  error_message?: string;
} 