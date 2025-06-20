import React, { useRef, useEffect, useState, useCallback } from 'react';

interface Annotation {
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

interface ImageCanvasProps {
  imageUrl: string;
  annotations: Annotation[];
  selectedTool: 'select' | 'bounding_box' | 'sam_point';
  onAnnotationCreate: (annotation: Omit<Annotation, 'id' | 'createdAt' | 'updatedAt'>) => void;
  onAnnotationUpdate: (id: string, annotation: Partial<Annotation>) => void;
  onAnnotationDelete: (id: string) => void;
  onSamPointClick?: (point: {x: number, y: number}, label: number) => void;
  samPoints?: Array<{x: number, y: number, label: number}>;
  maskPoints?: Array<{x: number, y: number}>;
  maskData?: string | null;
}

const ImageCanvas: React.FC<ImageCanvasProps> = ({
  imageUrl,
  annotations,
  selectedTool,
  onAnnotationCreate,
  onAnnotationUpdate,
  onAnnotationDelete,
  onSamPointClick,
  samPoints = [],
  maskPoints = [],
  maskData = null
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const maskImageRef = useRef<HTMLImageElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<{x: number, y: number} | null>(null);
  const [currentRect, setCurrentRect] = useState<{x: number, y: number, width: number, height: number} | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [maskLoaded, setMaskLoaded] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({x: 0, y: 0});

  // Load and setup image
  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      
      // Set canvas size to match image size (no scaling)
      canvas.width = img.width;
      canvas.height = img.height;
      
      // No scaling - display image at original size
      setScale(1);
      setOffset({ x: 0, y: 0 });
      setImageLoaded(true);
      
      drawCanvas();
    };
    
    img.src = imageUrl;
  }, [imageUrl]);

  // Load and setup mask image
  useEffect(() => {
    if (!maskData) {
      setMaskLoaded(false);
      maskImageRef.current = null;
      if (imageLoaded) {
        drawCanvas();
      }
      return;
    }

    const maskImg = new Image();
    maskImg.onload = () => {
      maskImageRef.current = maskImg;
      setMaskLoaded(true);
      if (imageLoaded) {
        drawCanvas();
      }
    };
    
    maskImg.onerror = () => {
      console.error('Failed to load mask image');
      setMaskLoaded(false);
      maskImageRef.current = null;
    };
    
    // Convert base64 to data URL
    maskImg.src = `data:image/png;base64,${maskData}`;
  }, [maskData, imageLoaded]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    const img = imageRef.current;
    const maskImg = maskImageRef.current;
    
    if (!canvas || !ctx || !img) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw image with full opacity
    ctx.globalAlpha = 1.0;
    ctx.drawImage(img, offset.x, offset.y, img.width * scale, img.height * scale);
    
    // Draw mask overlay if available
    if (maskImg && maskData) {
      // Create a blue-colored mask overlay
      ctx.globalAlpha = 0.5; // Semi-transparent overlay
      ctx.globalCompositeOperation = 'multiply';
      
      // First draw the mask
      ctx.drawImage(maskImg, offset.x, offset.y, img.width * scale, img.height * scale);
      
      // Apply blue color overlay
      ctx.globalCompositeOperation = 'multiply';
      ctx.fillStyle = 'rgba(144, 238, 144, 1)'; // Light green color
      ctx.fillRect(offset.x, offset.y, img.width * scale, img.height * scale);
      
      // Reset composite operation and opacity
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1.0;
    }
    
    // Draw existing annotations
    annotations.forEach((annotation) => {
      if (annotation.type === 'bounding_box') {
        const { x, y, width = 0, height = 0 } = annotation.coordinates;
        
        // Convert image coordinates to canvas coordinates
        const canvasX = offset.x + x * scale;
        const canvasY = offset.y + y * scale;
        const canvasWidth = width * scale;
        const canvasHeight = height * scale;
        
        // Draw bounding box
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 2;
        ctx.strokeRect(canvasX, canvasY, canvasWidth, canvasHeight);
        
        // Draw label
        ctx.fillStyle = '#ff0000';
        ctx.font = '12px Arial';
        ctx.fillText(annotation.label, canvasX, canvasY - 5);
      } else if (annotation.type === 'segmentation' && annotation.coordinates.points) {
        // Draw segmentation as polygon
        const points = annotation.coordinates.points;
        if (points.length > 2) {
          ctx.beginPath();
          const firstPoint = points[0];
          ctx.moveTo(offset.x + firstPoint.x * scale, offset.y + firstPoint.y * scale);
          
          for (let i = 1; i < points.length; i++) {
            const point = points[i];
            ctx.lineTo(offset.x + point.x * scale, offset.y + point.y * scale);
          }
          
          ctx.closePath();
          ctx.fillStyle = 'rgba(255, 255, 0, 0.3)'; // Semi-transparent yellow
          ctx.fill();
          ctx.strokeStyle = '#ffeb3b';
          ctx.lineWidth = 2;
          ctx.stroke();
          
          // Draw label
          ctx.fillStyle = '#ffeb3b';
          ctx.font = '12px Arial';
          ctx.fillText(annotation.label, offset.x + firstPoint.x * scale, offset.y + firstPoint.y * scale - 5);
        }
      }
    });
    
    // Draw SAM points
    samPoints.forEach((samPoint, index) => {
      const canvasX = offset.x + samPoint.x * scale;
      const canvasY = offset.y + samPoint.y * scale;
      
      // Draw point circle
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 6, 0, 2 * Math.PI);
      ctx.fillStyle = samPoint.label === 1 ? '#4caf50' : '#f44336'; // Green for positive, red for negative
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw point number
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText((index + 1).toString(), canvasX, canvasY + 3);
      ctx.textAlign = 'start'; // Reset alignment
    });
    
    // Draw mask points
    maskPoints.forEach((maskPoint) => {
      const canvasX = offset.x + maskPoint.x * scale;
      const canvasY = offset.y + maskPoint.y * scale;
      
      // Draw point circle
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 1, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255, 0, 0, 0.1)'; // Semi-transparent green
      ctx.fill();
    });
    
    // Draw current drawing rectangle
    if (currentRect) {
      const { x, y, width, height } = currentRect;
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);
    }
  }, [annotations, currentRect, scale, offset, samPoints, maskPoints, maskData]);

  // Redraw canvas when annotations, samPoints, or maskPoints change
  useEffect(() => {
    if (imageLoaded) {
      drawCanvas();
    }
  }, [imageLoaded, drawCanvas]);

  const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const canvasToImageCoords = (canvasX: number, canvasY: number) => {
    return {
      x: (canvasX - offset.x) / scale,
      y: (canvasY - offset.y) / scale
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedTool !== 'bounding_box') return;
    
    const mousePos = getMousePos(e);
    setIsDrawing(true);
    setStartPoint(mousePos);
    setCurrentRect(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint || selectedTool !== 'bounding_box') return;
    
    const mousePos = getMousePos(e);
    const width = mousePos.x - startPoint.x;
    const height = mousePos.y - startPoint.y;
    
    setCurrentRect({
      x: width >= 0 ? startPoint.x : mousePos.x,
      y: height >= 0 ? startPoint.y : mousePos.y,
      width: Math.abs(width),
      height: Math.abs(height)
    });
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !startPoint || !currentRect || selectedTool !== 'bounding_box') return;
    
    // Minimum rectangle size
    if (currentRect.width < 10 || currentRect.height < 10) {
      setIsDrawing(false);
      setStartPoint(null);
      setCurrentRect(null);
      return;
    }
    
    // Convert canvas coordinates to image coordinates
    const imageStart = canvasToImageCoords(currentRect.x, currentRect.y);
    const imageEnd = canvasToImageCoords(currentRect.x + currentRect.width, currentRect.y + currentRect.height);
    
    // Create annotation
    const annotation = {
      imageId: 'current-image',
      type: 'bounding_box' as const,
      coordinates: {
        x: Math.min(imageStart.x, imageEnd.x),
        y: Math.min(imageStart.y, imageEnd.y),
        width: Math.abs(imageEnd.x - imageStart.x),
        height: Math.abs(imageEnd.y - imageStart.y)
      },
      label: `bbox-${annotations.length + 1}`
    };
    
    onAnnotationCreate(annotation);
    
    // Reset drawing state
    setIsDrawing(false);
    setStartPoint(null);
    setCurrentRect(null);
  };

  const handleAnnotationClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedTool !== 'select') return;
    
    const mousePos = getMousePos(e);
    const imagePos = canvasToImageCoords(mousePos.x, mousePos.y);
    
    // Find clicked annotation
    const clickedAnnotation = annotations.find(annotation => {
      if (annotation.type === 'bounding_box') {
        const { x, y, width = 0, height = 0 } = annotation.coordinates;
        return imagePos.x >= x && imagePos.x <= x + width &&
               imagePos.y >= y && imagePos.y <= y + height;
      }
      return false;
    });
    
    if (clickedAnnotation) {
      // For now, just log the clicked annotation
      console.log('Clicked annotation:', clickedAnnotation);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedTool === 'select') {
      handleAnnotationClick(e);
    } else if (selectedTool === 'sam_point') {
      const mousePos = getMousePos(e);
      const imagePos = canvasToImageCoords(mousePos.x, mousePos.y);
      
      // Determine if this is a positive or negative point (shift for negative)
      const label = e.shiftKey ? 0 : 1; // 1 for positive, 0 for negative
      
      console.log('SAM point clicked at:', imagePos, 'label:', label);
      
      if (onSamPointClick) {
        onSamPointClick(imagePos, label);
      }
    }
  };

  return (
    <div style={{ 
      padding: '1rem',
      minHeight: '100%',
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem'
    }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'flex-start',
        alignItems: 'flex-start'
      }}>
        <canvas
          ref={canvasRef}
          style={{
            border: '2px solid #ddd',
            borderRadius: '8px',
            backgroundColor: '#fff',
            cursor: selectedTool === 'bounding_box' ? 'crosshair' : 
                    selectedTool === 'sam_point' ? 'pointer' : 'default',
            maxWidth: 'none',
            display: 'block'
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleCanvasClick}
        />
      </div>
      <div style={{
        textAlign: 'center',
        color: '#666',
        backgroundColor: '#fff',
        padding: '0.75rem',
        borderRadius: '8px',
        border: '1px solid #ddd'
      }}>
        <h3 style={{ margin: '0 0 0.5rem 0' }}>Interactive Canvas</h3>
        <p style={{ margin: '0 0 0.5rem 0' }}>
          Selected Tool: <strong>{selectedTool.replace('_', ' ').toUpperCase()}</strong>
        </p>
        {selectedTool === 'bounding_box' && (
          <p style={{ margin: '0', fontSize: '0.9rem', color: '#555' }}>
            Click and drag to draw a bounding box
          </p>
        )}
        {selectedTool === 'sam_point' && (
          <p style={{ margin: '0', fontSize: '0.9rem', color: '#555' }}>
            Click to add positive points, Shift+Click for negative points
          </p>
        )}
        <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem', color: '#666' }}>
          Annotations: {annotations.length}
        </p>
      </div>
    </div>
  );
};

export default ImageCanvas; 