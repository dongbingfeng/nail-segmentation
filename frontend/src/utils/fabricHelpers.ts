// @ts-ignore
import { fabric } from 'fabric';
import { Annotation } from '../types/annotations';

/**
 * Convert Fabric.js object to annotation format
 */
export const fabricObjectToAnnotation = (
  fabricObject: fabric.Object,
  imageId: string,
  label: string = 'nail'
): Omit<Annotation, 'id' | 'createdAt' | 'updatedAt'> => {
  if (fabricObject instanceof fabric.Rect) {
    return {
      imageId,
      type: 'bounding_box',
      coordinates: {
        x: fabricObject.left || 0,
        y: fabricObject.top || 0,
        width: fabricObject.width || 0,
        height: fabricObject.height || 0,
      },
      label,
    };
  }
  
  // For other object types, return basic coordinates
  return {
    imageId,
    type: 'bounding_box',
    coordinates: {
      x: fabricObject.left || 0,
      y: fabricObject.top || 0,
      width: fabricObject.width || 0,
      height: fabricObject.height || 0,
    },
    label,
  };
};

/**
 * Convert annotation to Fabric.js object
 */
export const annotationToFabricObject = (annotation: Annotation): fabric.Object => {
  if (annotation.type === 'bounding_box') {
    return new fabric.Rect({
      left: annotation.coordinates.x,
      top: annotation.coordinates.y,
      width: annotation.coordinates.width || 0,
      height: annotation.coordinates.height || 0,
      fill: 'transparent',
      stroke: '#00ff00',
      strokeWidth: 2,
      selectable: true,
      evented: true,
      name: `annotation-${annotation.id}`,
    });
  } else if (annotation.type === 'segmentation' && annotation.coordinates.points) {
    const points = annotation.coordinates.points.map(p => new fabric.Point(p.x, p.y));
    return new fabric.Polygon(points, {
      fill: 'rgba(0, 255, 0, 0.3)',
      stroke: '#00ff00',
      strokeWidth: 2,
      selectable: true,
      evented: true,
      name: `annotation-${annotation.id}`,
    });
  }
  
  // Fallback to rectangle
  return new fabric.Rect({
    left: annotation.coordinates.x,
    top: annotation.coordinates.y,
    width: 50,
    height: 50,
    fill: 'transparent',
    stroke: '#ff0000',
    strokeWidth: 2,
    selectable: true,
    evented: true,
    name: `annotation-${annotation.id}`,
  });
};

/**
 * Scale coordinates relative to image size
 */
export const scaleCoordinates = (
  coordinates: { x: number; y: number; width?: number; height?: number },
  fromSize: { width: number; height: number },
  toSize: { width: number; height: number }
) => {
  const scaleX = toSize.width / fromSize.width;
  const scaleY = toSize.height / fromSize.height;
  
  return {
    x: coordinates.x * scaleX,
    y: coordinates.y * scaleY,
    width: coordinates.width ? coordinates.width * scaleX : undefined,
    height: coordinates.height ? coordinates.height * scaleY : undefined,
  };
};

/**
 * Get image bounds within canvas
 */
export const getImageBounds = (canvas: fabric.Canvas): { left: number; top: number; width: number; height: number } | null => {
  const objects = canvas.getObjects();
  const imageObject = objects.find(obj => obj.type === 'image');
  
  if (imageObject) {
    return {
      left: imageObject.left || 0,
      top: imageObject.top || 0,
      width: (imageObject.width || 0) * (imageObject.scaleX || 1),
      height: (imageObject.height || 0) * (imageObject.scaleY || 1),
    };
  }
  
  return null;
};

/**
 * Constrain coordinates to image bounds
 */
export const constrainToImageBounds = (
  coordinates: { x: number; y: number; width?: number; height?: number },
  imageBounds: { left: number; top: number; width: number; height: number }
) => {
  const constrainedX = Math.max(imageBounds.left, Math.min(coordinates.x, imageBounds.left + imageBounds.width));
  const constrainedY = Math.max(imageBounds.top, Math.min(coordinates.y, imageBounds.top + imageBounds.height));
  
  let constrainedWidth = coordinates.width;
  let constrainedHeight = coordinates.height;
  
  if (coordinates.width) {
    constrainedWidth = Math.min(
      coordinates.width,
      imageBounds.left + imageBounds.width - constrainedX
    );
  }
  
  if (coordinates.height) {
    constrainedHeight = Math.min(
      coordinates.height,
      imageBounds.top + imageBounds.height - constrainedY
    );
  }
  
  return {
    x: constrainedX,
    y: constrainedY,
    width: constrainedWidth,
    height: constrainedHeight,
  };
};

/**
 * Clear all annotations from canvas
 */
export const clearAnnotations = (canvas: fabric.Canvas) => {
  const objects = canvas.getObjects();
  const annotationObjects = objects.filter(obj => 
    obj.name?.startsWith('annotation-') || obj.name === 'temp-rectangle'
  );
  
  annotationObjects.forEach(obj => canvas.remove(obj));
  canvas.renderAll();
};

/**
 * Set canvas cursor based on selected tool
 */
export const setCanvasCursor = (canvas: fabric.Canvas, tool: string) => {
  switch (tool) {
    case 'select':
      canvas.defaultCursor = 'default';
      canvas.hoverCursor = 'move';
      break;
    case 'bounding_box':
      canvas.defaultCursor = 'crosshair';
      canvas.hoverCursor = 'crosshair';
      break;
    case 'sam_point':
      canvas.defaultCursor = 'pointer';
      canvas.hoverCursor = 'pointer';
      break;
    default:
      canvas.defaultCursor = 'default';
      canvas.hoverCursor = 'default';
  }
};

/**
 * Export canvas as image data URL
 */
export const exportCanvasAsImage = (canvas: fabric.Canvas, format: string = 'png'): string => {
  return canvas.toDataURL({
    format,
    quality: 1,
  });
};

/**
 * Load image into canvas with proper scaling
 */
export const loadImageToCanvas = (
  canvas: fabric.Canvas,
  imageUrl: string,
  callback?: () => void
): Promise<fabric.Image> => {
  return new Promise((resolve, reject) => {
    fabric.Image.fromURL(
      imageUrl,
      (img) => {
        if (img) {
          // Scale image to fit canvas while maintaining aspect ratio
          const canvasWidth = canvas.getWidth();
          const canvasHeight = canvas.getHeight();
          const imgWidth = img.width || 1;
          const imgHeight = img.height || 1;
          
          const scale = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight);
          
          img.set({
            left: (canvasWidth - imgWidth * scale) / 2,
            top: (canvasHeight - imgHeight * scale) / 2,
            scaleX: scale,
            scaleY: scale,
            selectable: false,
            evented: false,
            name: 'background-image',
          });

          canvas.clear();
          canvas.add(img);
          canvas.renderAll();
          
          if (callback) callback();
          resolve(img);
        } else {
          reject(new Error('Failed to load image'));
        }
      },
      { crossOrigin: 'anonymous' }
    );
  });
}; 