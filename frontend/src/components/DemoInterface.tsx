import React, { useState } from 'react';
import { useImageNavigation } from '../contexts/ImageNavigationContext';
import ImageCanvas from './ImageCanvas';
import unetService from '../services/unetService';

const DemoInterface: React.FC = () => {
  const { currentImageData, isLoading, error } = useImageNavigation();
  const [maskData, setMaskData] = useState<string | null>(null);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [segmentationError, setSegmentationError] = useState<string | null>(null);
  const [segmentationTime, setSegmentationTime] = useState<number | null>(null);

  const handleSegmentImage = async () => {
    if (!currentImageData) {
      setSegmentationError('No image loaded');
      return;
    }

    setIsSegmenting(true);
    setSegmentationError(null);
    setSegmentationTime(null);

    try {
      // Convert image URL to base64
      const base64Image = await unetService.imageUrlToBase64(currentImageData.url);
      
      // Call U-Net segmentation API
      const response = await unetService.segmentImage({
        image_data: base64Image,
        threshold: 0.5,
        return_visualizations: false
      });

      if (response.success && response.mask_data) {
        setMaskData(response.mask_data);
        setSegmentationTime(response.processing_time_ms || null);
      } else {
        throw new Error('Segmentation failed: No mask data returned');
      }
    } catch (error) {
      console.error('Segmentation error:', error);
      setSegmentationError(
        error instanceof Error ? error.message : 'Segmentation failed'
      );
    } finally {
      setIsSegmenting(false);
    }
  };

  const clearMask = () => {
    setMaskData(null);
    setSegmentationError(null);
    setSegmentationTime(null);
  };

  if (isLoading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '400px',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            width: '40px', 
            height: '40px', 
            border: '4px solid #f3f3f3',
            borderTop: '4px solid #007bff',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 1rem'
          }} />
          <p style={{ color: '#666', margin: 0 }}>Loading image...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        padding: '2rem',
        backgroundColor: '#f8d7da',
        border: '1px solid #f5c6cb',
        borderRadius: '8px',
        color: '#721c24'
      }}>
        <h3 style={{ margin: '0 0 1rem 0' }}>Error Loading Image</h3>
        <p style={{ margin: 0 }}>{error}</p>
      </div>
    );
  }

  if (!currentImageData) {
    return (
      <div style={{ 
        padding: '2rem',
        backgroundColor: '#fff3cd',
        border: '1px solid #ffeaa7',
        borderRadius: '8px',
        color: '#856404',
        textAlign: 'center'
      }}>
        <h3 style={{ margin: '0 0 1rem 0' }}>No Image Available</h3>
        <p style={{ margin: 0 }}>Please ensure images are loaded in the system.</p>
      </div>
    );
  }

  return (
    <div style={{ 
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
      padding: '1rem'
    }}>
      {/* Control Panel */}
      <div style={{
        backgroundColor: '#fff',
        padding: '1rem',
        borderRadius: '8px',
        border: '1px solid #ddd',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        <div>
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#333' }}>
            {currentImageData.filename}
          </h3>
          <p style={{ margin: 0, color: '#666', fontSize: '0.9rem' }}>
            Size: {currentImageData.width} × {currentImageData.height}px
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <button
            onClick={handleSegmentImage}
            disabled={isSegmenting}
            style={{
              padding: '0.75rem 1.5rem',
              backgroundColor: isSegmenting ? '#6c757d' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: isSegmenting ? 'not-allowed' : 'pointer',
              fontSize: '1rem',
              fontWeight: '500',
              transition: 'background-color 0.2s'
            }}
          >
            {isSegmenting ? 'Segmenting...' : 'Segment Image'}
          </button>
          
          {maskData && (
            <button
              onClick={clearMask}
              style={{
                padding: '0.75rem 1rem',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '1rem'
              }}
            >
              Clear Mask
            </button>
          )}
        </div>
      </div>

      {/* Status Messages */}
      {segmentationError && (
        <div style={{
          padding: '1rem',
          backgroundColor: '#f8d7da',
          border: '1px solid #f5c6cb',
          borderRadius: '8px',
          color: '#721c24'
        }}>
          <strong>Segmentation Error:</strong> {segmentationError}
        </div>
      )}

      {maskData && segmentationTime && (
        <div style={{
          padding: '1rem',
          backgroundColor: '#d4edda',
          border: '1px solid #c3e6cb',
          borderRadius: '8px',
          color: '#155724'
        }}>
          <strong>Segmentation Complete!</strong> Processing time: {segmentationTime.toFixed(1)}ms
        </div>
      )}

      {/* Image Canvas */}
      <div style={{ 
        backgroundColor: '#fff',
        borderRadius: '8px',
        border: '1px solid #ddd',
        overflow: 'hidden'
      }}>
        <ImageCanvas
          imageUrl={currentImageData.url}
          annotations={[]}
          selectedTool="select"
          onAnnotationCreate={() => {}}
          onAnnotationUpdate={() => {}}
          onAnnotationDelete={() => {}}
          maskData={maskData}
        />
      </div>

      {/* Instructions */}
      <div style={{
        backgroundColor: '#f8f9fa',
        padding: '1rem',
        borderRadius: '8px',
        border: '1px solid #dee2e6'
      }}>
        <h4 style={{ margin: '0 0 0.5rem 0', color: '#495057' }}>How to use:</h4>
        <ol style={{ margin: 0, paddingLeft: '1.2rem', color: '#6c757d' }}>
          <li>Use the navigation controls to browse through images</li>
          <li>Click "Segment Image" to run nail segmentation on the current image</li>
          <li>The segmentation mask will appear as a semi-transparent overlay</li>
          <li>Click "Clear Mask" to remove the overlay and try again</li>
        </ol>
      </div>
    </div>
  );
};

export default DemoInterface; 