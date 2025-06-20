import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ImageData, Annotation } from '../types/annotations';
import { imageService } from '../services/imageService';

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

interface ImageNavigationContextType {
  images: ImageMetadata[];
  currentImageIndex: number;
  currentImage: ImageMetadata | null;
  currentImageData: ImageData | null;
  isLoading: boolean;
  error: string | null;
  progress: ProgressStatus | null;
  annotations: Annotation[];
  navigateNext: () => Promise<void>;
  navigatePrevious: () => Promise<void>;
  navigateToIndex: (index: number) => Promise<void>;
  markImageComplete: (imageId: string) => Promise<void>;
  saveCurrentAnnotations: (annotations: Annotation[]) => Promise<void>;
  updateAnnotations: (annotations: Annotation[]) => void;
  refreshImageList: () => Promise<void>;
}

const ImageNavigationContext = createContext<ImageNavigationContextType | undefined>(undefined);

interface ImageNavigationProviderProps {
  children: ReactNode;
}

export const ImageNavigationProvider: React.FC<ImageNavigationProviderProps> = ({ children }) => {
  const [images, setImages] = useState<ImageMetadata[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState<number>(0);
  const [currentImageData, setCurrentImageData] = useState<ImageData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressStatus | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);

  const currentImage = images[currentImageIndex] || null;

  // Initialize: Load image list and progress
  useEffect(() => {
    initializeImageNavigation();
  }, []);

  // Load annotations when current image changes
  useEffect(() => {
    if (currentImage) {
      loadCurrentImageData();
    }
  }, [currentImage]);

  const initializeImageNavigation = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Load image list
      const imageList = await imageService.fetchImageList();
      setImages(imageList);

      // Load progress to determine starting index
      const progressData = await imageService.fetchLabelingProgress();
      setProgress(progressData);
      
      const startIndex = Math.min(progressData.currentImageIndex, Math.max(0, imageList.length - 1));
      setCurrentImageIndex(startIndex);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize image navigation');
      console.error('Error initializing image navigation:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadCurrentImageData = async () => {
    if (!currentImage) {
      setCurrentImageData(null);
      setAnnotations([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Load annotations for current image
      const imageAnnotations = await imageService.fetchImageAnnotations(currentImage.id);
      setAnnotations(imageAnnotations);

      // Create ImageData object for compatibility with existing components
      const imageData: ImageData = {
        id: currentImage.id,
        filename: currentImage.filename,
        url: currentImage.url,
        width: currentImage.width,
        height: currentImage.height,
        annotations: imageAnnotations,
        uploadedAt: currentImage.lastModified,
      };

      setCurrentImageData(imageData);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load image data');
      console.error('Error loading image data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const navigateNext = async () => {
    if (currentImageIndex < images.length - 1) {
      await navigateToIndex(currentImageIndex + 1);
    }
  };

  const navigatePrevious = async () => {
    if (currentImageIndex > 0) {
      await navigateToIndex(currentImageIndex - 1);
    }
  };

  const navigateToIndex = async (index: number) => {
    if (index < 0 || index >= images.length) {
      return;
    }

    // Save current annotations before navigation
    if (currentImage && annotations.length > 0) {
      await saveCurrentAnnotations(annotations);
    }

    // Update progress index on backend
    try {
      await imageService.updateProgressIndex(index);
    } catch (err) {
      console.warn('Failed to update progress index:', err);
    }

    setCurrentImageIndex(index);

    // Refresh progress data
    try {
      const progressData = await imageService.fetchLabelingProgress();
      setProgress(progressData);
    } catch (err) {
      console.warn('Failed to refresh progress:', err);
    }
  };

  const markImageComplete = async (imageId: string) => {
    try {
      // Update the image's completion status locally
      setImages(prevImages => 
        prevImages.map(img => 
          img.id === imageId 
            ? { ...img, isCompleted: true, annotationCount: annotations.length }
            : img
        )
      );

      // Refresh progress data
      const progressData = await imageService.fetchLabelingProgress();
      setProgress(progressData);

    } catch (err) {
      console.error('Error marking image complete:', err);
    }
  };

  const saveCurrentAnnotations = async (annotationsToSave: Annotation[]) => {
    if (!currentImage) return;

    try {
      await imageService.saveImageAnnotations(currentImage.id, annotationsToSave);
      
      // Update local state
      setAnnotations(annotationsToSave);
      
      // Mark image as complete if it has annotations
      if (annotationsToSave.length > 0) {
        await markImageComplete(currentImage.id);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save annotations');
      console.error('Error saving annotations:', err);
    }
  };

  const updateAnnotations = (newAnnotations: Annotation[]) => {
    setAnnotations(newAnnotations);
  };

  const refreshImageList = async () => {
    try {
      const imageList = await imageService.fetchImageList();
      setImages(imageList);

      const progressData = await imageService.fetchLabelingProgress();
      setProgress(progressData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh image list');
    }
  };

  const contextValue: ImageNavigationContextType = {
    images,
    currentImageIndex,
    currentImage,
    currentImageData,
    isLoading,
    error,
    progress,
    annotations,
    navigateNext,
    navigatePrevious,
    navigateToIndex,
    markImageComplete,
    saveCurrentAnnotations,
    updateAnnotations,
    refreshImageList,
  };

  return (
    <ImageNavigationContext.Provider value={contextValue}>
      {children}
    </ImageNavigationContext.Provider>
  );
};

export const useImageNavigation = (): ImageNavigationContextType => {
  const context = useContext(ImageNavigationContext);
  if (context === undefined) {
    throw new Error('useImageNavigation must be used within an ImageNavigationProvider');
  }
  return context;
};

export default ImageNavigationContext; 