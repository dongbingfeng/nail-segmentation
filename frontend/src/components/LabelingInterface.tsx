import React, { useState, useEffect } from 'react';
import ImageCanvas from './ImageCanvas';
import ToolPalette from './ToolPalette';
import { useImageNavigation } from '../contexts/ImageNavigationContext';
import { Annotation, SAMRequest } from '../types/annotations';
import { samService } from '../services/samService';

const LabelingInterface: React.FC = () => {
  const [selectedTool, setSelectedTool] = useState<'select' | 'bounding_box' | 'sam_point'>('select');
  const [isProcessing, setIsProcessing] = useState(false);
  const [samPoints, setSamPoints] = useState<Array<{x: number, y: number, label: number}>>([]);
  const [maskPoints, setMaskPoints] = useState<Array<{x: number, y: number}>>([]);
  
  const {
    currentImageData,
    annotations,
    updateAnnotations,
    saveCurrentAnnotations,
    isLoading,
    error
  } = useImageNavigation();

  // No auto-save - annotations are only saved when "Save Label" button is clicked

  const handleToolSelect = (tool: string) => {
    setSelectedTool(tool as 'select' | 'bounding_box' | 'sam_point');
  };

  const handleAnnotationCreate = async (annotationData: Omit<Annotation, 'id' | 'createdAt' | 'updatedAt'>) => {
    if (!currentImageData) return;
    
    try {
      // Generate temporary ID for now - will be replaced with API response
      const newAnnotation: Annotation = {
        ...annotationData,
        id: `temp-${Date.now()}`,
        imageId: currentImageData.id,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      const updatedAnnotations = [...annotations, newAnnotation];
      updateAnnotations(updatedAnnotations);
      
      console.log('Created annotation:', newAnnotation);
    } catch (err) {
      console.error('Error creating annotation:', err);
    }
  };

  const handleAnnotationUpdate = async (id: string, updates: Partial<Annotation>) => {
    try {
      const updatedAnnotations = annotations.map(annotation => 
        annotation.id === id 
          ? { ...annotation, ...updates, updatedAt: new Date().toISOString() }
          : annotation
      );
      
      updateAnnotations(updatedAnnotations);
      console.log('Updated annotation:', id, updates);
    } catch (err) {
      console.error('Error updating annotation:', err);
    }
  };

  const handleAnnotationDelete = async (id: string) => {
    try {
      const updatedAnnotations = annotations.filter(annotation => annotation.id !== id);
      updateAnnotations(updatedAnnotations);
      
      console.log('Deleted annotation:', id);
    } catch (err) {
      console.error('Error deleting annotation:', err);
    }
  };

  const handleSamPointClick = (point: {x: number, y: number}, label: number = 1) => {
    // Add point to SAM points collection
    setSamPoints(prev => [...prev, { ...point, label }]);
  };

  const handleClearSamPoints = () => {
    setSamPoints([]);
    setMaskPoints([]);
  };

  const handleClearMaskPoints = () => {
    setMaskPoints([]);
    // Also remove all bounding box annotations
    const updatedAnnotations = annotations.filter(annotation => annotation.type !== 'bounding_box');
    updateAnnotations(updatedAnnotations);
  };

  const handleSaveLabel = async () => {
    if (!currentImageData) {
      console.warn('No image data available to save');
      return;
    }

    try {
      let updatedAnnotations = [...annotations];

      // If mask points exist, convert them to a segmentation annotation
      if (maskPoints.length > 0) {
        // Calculate bounding box from mask points
        const minX = Math.min(...maskPoints.map(p => p.x));
        const minY = Math.min(...maskPoints.map(p => p.y));
        const maxX = Math.max(...maskPoints.map(p => p.x));
        const maxY = Math.max(...maskPoints.map(p => p.y));

        // Create segmentation annotation from mask points
        const maskAnnotation: Annotation = {
          id: `mask-${Date.now()}`,
          imageId: currentImageData.id,
          type: 'segmentation',
          coordinates: {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY,
            points: maskPoints.map(p => ({ x: p.x, y: p.y }))
          },
          label: `mask-label-${annotations.filter(a => a.type === 'segmentation').length + 1}`,
          confidence: 1.0, // Set high confidence since this is manually saved
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };

        // Add mask annotation to the list
        updatedAnnotations = [...updatedAnnotations, maskAnnotation];
        updateAnnotations(updatedAnnotations);

        // Clear mask points after converting to annotation
        setMaskPoints([]);
      }

      // Save all annotations to backend (including bounding boxes and segmentations)
      await saveCurrentAnnotations(updatedAnnotations);
      
      console.log('All annotations saved to backend:', updatedAnnotations);
    } catch (err) {
      console.error('Error saving annotations:', err);
      alert(`Failed to save annotations: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const handleSamProcess = async () => {
    if (!currentImageData) {
      console.warn('No image or SAM points available');
      return;
    }
    if (samPoints.length === 0) {
      setSamPoints([]);
    }
    // Find a bounding box from existing annotations or create a default one
    let boundingBox = {
      topLeft: { x: 0, y: 0 },
      bottomRight: { x: currentImageData.width, y: currentImageData.height }
    };

    // Use the first bounding box annotation if available
    const bboxAnnotation = annotations.find(ann => ann.type === 'bounding_box');
    if (bboxAnnotation && bboxAnnotation.coordinates.width && bboxAnnotation.coordinates.height) {
      boundingBox = {
        topLeft: { 
          x: bboxAnnotation.coordinates.x, 
          y: bboxAnnotation.coordinates.y 
        },
        bottomRight: { 
          x: bboxAnnotation.coordinates.x + bboxAnnotation.coordinates.width,
          y: bboxAnnotation.coordinates.y + bboxAnnotation.coordinates.height
        }
      };
    }

    setIsProcessing(true);
    try {
      console.log('Processing with SAM...', { 
        imageId: currentImageData.id, 
        boundingBox,
        points: samPoints 
      });
      
      // Prepare SAM request with new interface
      const samRequest: SAMRequest = {
        imageId: currentImageData.id,
        boundingBox,
        points: samPoints.map(p => ({ x: p.x, y: p.y })),
        labels: samPoints.map(p => p.label)
      };

      // Call real SAM service
      const result = await samService.processSegmentation(samRequest);
      if (result.success) {
        // Handle mask_points if they exist in the response
        if (result.mask_points && result.mask_points.length > 0) {
          setMaskPoints(result.mask_points);
          
        } else {
          setMaskPoints([]); // Clear existing mask points if none received
        }
        
        // Convert SAM masks to annotations if they exist
        if (result.masks.length > 0) {
          result.masks.forEach((mask, index) => {
            const samAnnotation: Annotation = {
              id: `sam-${Date.now()}-${index}`,
              imageId: currentImageData.id,
              type: 'segmentation',
              coordinates: {
                x: Math.min(...mask.points.map(p => p.x)),
                y: Math.min(...mask.points.map(p => p.y)),
                points: mask.points,
              },
              label: `sam-nail-${annotations.length + index + 1}`,
              confidence: mask.confidence,
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
            };
            
            const updatedAnnotations = [...annotations, samAnnotation];
            updateAnnotations(updatedAnnotations);
          });
        }

        // Clear SAM points after successful processing
        setSamPoints([]);
        console.log('SAM processing completed successfully');
      } else {
        // Clear mask points on failure
        setMaskPoints([]);
        console.error('SAM processing failed:', result.error);
        alert(`SAM processing failed: ${result.error || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error processing with SAM:', err);
      alert(`SAM processing error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  if (isLoading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '400px',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px'
      }}>
        <div>Loading image...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center',
        backgroundColor: '#ffebee',
        color: '#c62828',
        borderRadius: '8px',
        border: '1px solid #ef5350'
      }}>
        <h3>Error</h3>
        <p>{error}</p>
        <button 
          onClick={() => window.location.reload()}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: '#1976d2',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (!currentImageData) {
    return (
      <div style={{ 
        padding: '2rem', 
        textAlign: 'center',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px'
      }}>
        No image data available
      </div>
    );
  }

  return (
    <div style={{ 
      display: 'flex', 
      gap: '1rem', 
      padding: '1rem',
      height: '100vh',
      overflow: 'hidden'
    }}>
      {/* Left Panel - Tools */}
      <div style={{ 
        width: '300px', 
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        height: '100%'
      }}>
        <ToolPalette
          selectedTool={selectedTool}
          onToolSelect={handleToolSelect}
          onSamProcess={handleSamProcess}
          onClearSamPoints={handleClearSamPoints}
          onClearMaskPoints={handleClearMaskPoints}
          onSaveLabel={handleSaveLabel}
          isProcessing={isProcessing}
          samPointsCount={samPoints.length}
          maskPointsCount={maskPoints.length}
          boundingBoxCount={annotations.filter(a => a.type === 'bounding_box').length}
          totalAnnotationCount={annotations.length}
        />
        
        {/* Annotations List */}
        <div style={{
          backgroundColor: '#f8f9fa',
          borderRadius: '8px',
          padding: '1rem',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          flex: 1,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column'
        }}>
          <h3 style={{ 
            margin: '0 0 1rem 0', 
            fontSize: '1.1rem', 
            color: '#333' 
          }}>
            Annotations ({annotations.length})
          </h3>
          
          {annotations.length === 0 ? (
            <p style={{ 
              color: '#666', 
              fontSize: '0.9rem',
              margin: 0 
            }}>
              No annotations yet. Use the tools above to create some.
            </p>
          ) : (
            <div style={{ 
              flex: 1,
              overflowY: 'auto',
              minHeight: 0
            }}>
              {annotations.map((annotation, index) => (
                <div 
                  key={annotation.id}
                  style={{
                    padding: '0.5rem',
                    marginBottom: '0.5rem',
                    backgroundColor: '#fff',
                    borderRadius: '4px',
                    border: '1px solid #ddd',
                    fontSize: '0.9rem'
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>
                    {annotation.type === 'bounding_box' ? '📦' : '🎯'} {annotation.label}
                  </div>
                  <div style={{ color: '#666', fontSize: '0.8rem' }}>
                    {annotation.type === 'bounding_box' 
                      ? `${Math.round(annotation.coordinates.x)}, ${Math.round(annotation.coordinates.y)} - ${Math.round(annotation.coordinates.width || 0)}×${Math.round(annotation.coordinates.height || 0)}`
                      : `${annotation.coordinates.points?.length || 0} points`
                    }
                    {annotation.confidence && (
                      <span> (conf: {Math.round(annotation.confidence * 100)}%)</span>
                    )}
                  </div>
                  <button
                    onClick={() => handleAnnotationDelete(annotation.id)}
                    style={{
                      marginTop: '0.25rem',
                      padding: '0.25rem 0.5rem',
                      backgroundColor: '#ff4444',
                      color: 'white',
                      border: 'none',
                      borderRadius: '3px',
                      fontSize: '0.8rem',
                      cursor: 'pointer'
                    }}
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right Panel - Canvas */}
      <div style={{ 
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        minWidth: 0
      }}>
        <div style={{
          flex: 1,
          overflow: 'auto',
          minHeight: 0,
          border: '1px solid #ddd',
          borderRadius: '8px',
          backgroundColor: '#f9f9f9'
        }}>
          <ImageCanvas
            imageUrl={currentImageData.url}
            annotations={annotations}
            selectedTool={selectedTool}
            onAnnotationCreate={handleAnnotationCreate}
            onAnnotationUpdate={handleAnnotationUpdate}
            onAnnotationDelete={handleAnnotationDelete}
            onSamPointClick={handleSamPointClick}
            samPoints={samPoints}
            maskPoints={maskPoints}
          />
        </div>
        <div style={{ 
          marginBottom: '1rem',
          padding: '0.75rem 1rem',
          backgroundColor: '#e3f2fd',
          borderRadius: '6px',
          border: '1px solid #1976d2',
          flexShrink: 0
        }}>
          <h2 style={{ margin: '0 0 0.5rem 0', color: '#1976d2' }}>
            {currentImageData.filename}
          </h2>
          <p style={{ margin: 0, fontSize: '0.9rem', color: '#1565c0' }}>
            Current tool: <strong>{selectedTool.replace('_', ' ').toUpperCase()}</strong> | Image: {currentImageData.width}×{currentImageData.height}px
          </p>
        </div>
      </div>
    </div>
  );
};

export default LabelingInterface; 