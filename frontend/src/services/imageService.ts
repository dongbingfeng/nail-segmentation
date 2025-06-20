import { Annotation } from '../types/annotations';

interface ImageMetadata {
  id: string;
  filename: string;
  url: string;
  width: number;
  height: number;
  isCompleted: boolean;
  annotationCount: number;
  lastModified: string;
}

interface ProgressStatus {
  totalImages: number;
  completedImages: number;
  currentImageIndex: number;
  percentComplete: number;
}

class ImageService {
  private baseUrl: string;

  constructor() {
    // Default to same origin, can be configured for different environments
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  async fetchImageList(): Promise<ImageMetadata[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/images`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch image list: ${response.status} ${response.statusText}`);
      }

      const images = await response.json();
      return images;
    } catch (error) {
      console.error('Error fetching image list:', error);
      throw new Error(`Failed to fetch image list: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async fetchImageAnnotations(imageId: string): Promise<Annotation[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/images/${imageId}/annotations`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch annotations for image ${imageId}: ${response.status} ${response.statusText}`);
      }

      const annotations = await response.json();
      return annotations;
    } catch (error) {
      console.error(`Error fetching annotations for image ${imageId}:`, error);
      throw new Error(`Failed to fetch annotations: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async saveImageAnnotations(imageId: string, annotations: Annotation[]): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/images/${imageId}/annotations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(annotations),
      });

      if (!response.ok) {
        throw new Error(`Failed to save annotations for image ${imageId}: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log(`Saved ${result.count} annotations for image ${imageId}`);
    } catch (error) {
      console.error(`Error saving annotations for image ${imageId}:`, error);
      throw new Error(`Failed to save annotations: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async fetchLabelingProgress(): Promise<ProgressStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/api/labeling/progress`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch labeling progress: ${response.status} ${response.statusText}`);
      }

      const progress = await response.json();
      return progress;
    } catch (error) {
      console.error('Error fetching labeling progress:', error);
      throw new Error(`Failed to fetch labeling progress: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async updateProgressIndex(currentIndex: number): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/labeling/progress/${currentIndex}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to update progress index: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log(`Updated progress index to ${result.currentIndex}`);
    } catch (error) {
      console.error('Error updating progress index:', error);
      throw new Error(`Failed to update progress index: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  // Health check method to verify backend connectivity
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }

  // Method to handle network errors gracefully
  private async handleResponse(response: Response): Promise<any> {
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      
      try {
        const errorBody = await response.json();
        if (errorBody.detail) {
          errorMessage = errorBody.detail;
        }
      } catch {
        // If we can't parse error body, use default message
      }
      
      throw new Error(errorMessage);
    }

    return await response.json();
  }
}

// Create and export singleton instance
export const imageService = new ImageService();

// Export types for use in other modules
export type { ImageMetadata, ProgressStatus }; 