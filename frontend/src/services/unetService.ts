/**
 * U-Net API Service
 * Handles communication with the backend U-Net segmentation endpoints
 */

import { UNetSegmentationRequest, UNetSegmentationResponse, UNetHealthResponse } from '../types/unet';

class UNetService {
  private baseUrl: string;

  constructor() {
    // Use the same base URL pattern as other services
    this.baseUrl = process.env.NODE_ENV === 'production' 
      ? '/api' 
      : 'http://localhost:8000/api';
  }

  /**
   * Segment an image using the U-Net model
   * @param request - The segmentation request containing base64 image data
   * @returns Promise resolving to segmentation response
   */
  async segmentImage(request: UNetSegmentationRequest): Promise<UNetSegmentationResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/unet/segment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const result: UNetSegmentationResponse = await response.json();
      
      if (!result.success) {
        throw new Error(result.error?.message || 'Segmentation failed');
      }

      return result;
    } catch (error) {
      console.error('UNet segmentation error:', error);
      throw new Error(
        error instanceof Error 
          ? `Segmentation failed: ${error.message}`
          : 'Segmentation failed: Unknown error'
      );
    }
  }

  /**
   * Check the health status of the U-Net service
   * @returns Promise resolving to health status
   */
  async checkHealth(): Promise<UNetHealthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/unet/health`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('UNet health check error:', error);
      throw new Error(
        error instanceof Error 
          ? `Health check failed: ${error.message}`
          : 'Health check failed: Unknown error'
      );
    }
  }

  /**
   * Convert an image file to base64 string
   * @param imageFile - The image file to convert
   * @returns Promise resolving to base64 string
   */
  async imageToBase64(imageFile: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove the data URL prefix (e.g., "data:image/png;base64,")
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = () => reject(new Error('Failed to read image file'));
      reader.readAsDataURL(imageFile);
    });
  }

  /**
   * Convert an image URL to base64 string
   * @param imageUrl - The image URL to convert
   * @returns Promise resolving to base64 string
   */
  async imageUrlToBase64(imageUrl: string): Promise<string> {
    try {
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch image: ${response.statusText}`);
      }
      
      const blob = await response.blob();
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result as string;
          // Remove the data URL prefix (e.g., "data:image/png;base64,")
          const base64 = result.split(',')[1];
          resolve(base64);
        };
        reader.onerror = () => reject(new Error('Failed to convert image to base64'));
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error('Image URL to base64 conversion error:', error);
      throw new Error(
        error instanceof Error 
          ? `Failed to convert image: ${error.message}`
          : 'Failed to convert image: Unknown error'
      );
    }
  }
}

// Export singleton instance
export const unetService = new UNetService();
export default unetService; 