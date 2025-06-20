import { SAMRequest, SAMResponse } from '../types/annotations';
import { SAMHealthResponse, SAMModelInfo, SAMInitializeResponse } from '../types/sam';

class SAMService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  async processSegmentation(request: SAMRequest): Promise<SAMResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/sam/segment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Transform backend response to match SAMResponse interface
      return {
        masks: data.masks || [],
        mask_points: data.mask_points || [],
        success: data.success,
        error: data.error,
        processing_time: data.processing_time
      };

    } catch (error) {
      console.error('SAM segmentation request failed:', error);
      return this.handleSAMError(error);
    }
  }

  async checkSAMHealth(): Promise<SAMHealthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/sam/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();

    } catch (error) {
      console.error('SAM health check failed:', error);
      return {
        status: "unhealthy",
        model_loaded: false,
        model_type: "unknown",
        device: "unknown",
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async getModelInfo(): Promise<SAMModelInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/api/sam/model-info`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();

    } catch (error) {
      console.error('SAM model info request failed:', error);
      // Return default values on error
      return {
        initialized: false,
        model_type: "unknown",
        device: "unknown",
        confidence_threshold: 0.5,
        max_points: 10
      };
    }
  }

  async initializeModel(): Promise<SAMInitializeResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/sam/initialize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();

    } catch (error) {
      console.error('SAM initialization failed:', error);
      return {
        message: error instanceof Error ? error.message : 'Initialization failed',
        success: false
      };
    }
  }

  private handleSAMError(error: any): SAMResponse {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    
    // Categorize errors for user-friendly messages
    let userMessage = errorMessage;
    
    if (errorMessage.includes('HTTP 400')) {
      userMessage = 'Invalid request. Please check your point selection.';
    } else if (errorMessage.includes('HTTP 500')) {
      userMessage = 'SAM processing failed. Try clicking on clearer object boundaries.';
    } else if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
      userMessage = 'Connection lost. Check network and retry.';
    } else if (errorMessage.includes('GPU') || errorMessage.includes('memory')) {
      userMessage = 'Processing with CPU mode. Performance may be slower.';
    }

    return {
      masks: [],
      success: false,
      error: userMessage
    };
  }

  // Utility method to test if SAM service is available
  async isAvailable(): Promise<boolean> {
    try {
      const health = await this.checkSAMHealth();
      return health.status !== "unhealthy";
    } catch {
      return false;
    }
  }

  // Utility method to get a human-readable status
  async getStatus(): Promise<string> {
    try {
      const health = await this.checkSAMHealth();
      
      if (health.status === "healthy") {
        return `SAM ready (${health.model_type} on ${health.device})`;
      } else if (health.status === "loading") {
        return "SAM initializing...";
      } else {
        return `SAM unavailable: ${health.error || 'Unknown error'}`;
      }
    } catch {
      return "SAM service unreachable";
    }
  }
}

export const samService = new SAMService(); 