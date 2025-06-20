export interface Annotation {
  id: string;
  imageId: string;
  type: 'bounding_box' | 'segmentation';
  coordinates: {
    x: number;
    y: number;
    width?: number;
    height?: number;
    points?: Array<{x: number, y: number}>;
  };
  label: string;
  confidence?: number;
  createdAt: string;
  updatedAt: string;
}

export interface ImageData {
  id: string;
  filename: string;
  url: string;
  width: number;
  height: number;
  annotations: Annotation[];
  uploadedAt: string;
}

export interface CreateAnnotationRequest {
  imageId: string;
  type: 'bounding_box' | 'segmentation';
  coordinates: {
    x: number;
    y: number;
    width?: number;
    height?: number;
    points?: Array<{x: number, y: number}>;
  };
  label: string;
}

export interface UpdateAnnotationRequest {
  coordinates?: {
    x: number;
    y: number;
    width?: number;
    height?: number;
    points?: Array<{x: number, y: number}>;
  };
  label?: string;
}

export interface SAMRequest {
  imageId: string;
  boundingBox: {
    topLeft: {x: number, y: number};
    bottomRight: {x: number, y: number};
  };
  points: Array<{x: number, y: number}>;
  labels: Array<number>; // 1 for positive, 0 for negative
}

export interface SAMResponse {
  masks: Array<{
    points: Array<{x: number, y: number}>;
    confidence: number;
  }>;
  mask_points?: Array<{x: number, y: number}>;
  success: boolean;
  error?: string;
  processing_time?: number;
}

export interface ImageMetadata {
  id: string;
  filename: string;
  url: string;
  width: number;
  height: number;
  isCompleted: boolean;
  annotationCount: number;
  lastModified: string;
}

export interface ProgressStatus {
  totalImages: number;
  completedImages: number;
  currentImageIndex: number;
  percentComplete: number;
}