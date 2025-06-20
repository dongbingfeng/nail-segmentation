import React from 'react';
import { useImageNavigation } from '../contexts/ImageNavigationContext';

interface NavigationControlsProps {
  className?: string;
}

const NavigationControls: React.FC<NavigationControlsProps> = ({ className = '' }) => {
  const {
    currentImageIndex,
    images,
    currentImage,
    progress,
    isLoading,
    navigateNext,
    navigatePrevious,
    navigateToIndex
  } = useImageNavigation();

  const canGoNext = currentImageIndex < images.length - 1;
  const canGoPrevious = currentImageIndex > 0;

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.code === 'Space' && canGoNext) {
      event.preventDefault();
      navigateNext();
    } else if (event.key === 'ArrowRight' && canGoNext) {
      event.preventDefault();
      navigateNext();
    } else if (event.key === 'ArrowLeft' && canGoPrevious) {
      event.preventDefault();
      navigatePrevious();
    }
  };

  const handleImageSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedIndex = parseInt(event.target.value);
    navigateToIndex(selectedIndex);
  };

  if (images.length === 0) {
    return (
      <div className={`navigation-controls ${className}`} style={{
        padding: '1rem',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px',
        border: '1px solid #ddd',
        textAlign: 'center'
      }}>
        <p style={{ margin: 0, color: '#666' }}>No images found for labeling</p>
      </div>
    );
  }

  return (
    <div 
      className={`navigation-controls ${className}`}
      style={{
        padding: '1rem',
        backgroundColor: '#fff',
        borderRadius: '8px',
        border: '1px solid #ddd',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      {/* Progress Information */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '1rem',
        padding: '0.5rem',
        backgroundColor: '#f8f9fa',
        borderRadius: '6px'
      }}>
        <div style={{ fontSize: '0.9rem', color: '#666' }}>
          <strong>Image {currentImageIndex + 1} of {images.length}</strong>
          {currentImage && (
            <span style={{ marginLeft: '0.5rem' }}>
              ({currentImage.filename})
            </span>
          )}
        </div>
        
        {progress && (
          <div style={{ fontSize: '0.9rem', color: '#28a745' }}>
            <strong>{progress.completedImages}/{progress.totalImages} completed</strong>
            <span style={{ marginLeft: '0.5rem' }}>
              ({progress.percentComplete}%)
            </span>
          </div>
        )}
      </div>

      {/* Navigation Controls */}
      <div style={{
        display: 'flex',
        gap: '1rem',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        {/* Previous Button */}
        <button
          onClick={navigatePrevious}
          disabled={!canGoPrevious || isLoading}
          style={{
            padding: '0.75rem 1.5rem',
            backgroundColor: canGoPrevious && !isLoading ? '#6c757d' : '#e9ecef',
            color: canGoPrevious && !isLoading ? '#fff' : '#6c757d',
            border: 'none',
            borderRadius: '6px',
            cursor: canGoPrevious && !isLoading ? 'pointer' : 'not-allowed',
            fontSize: '1rem',
            fontWeight: 'bold',
            transition: 'all 0.2s ease',
            minWidth: '120px'
          }}
          onMouseEnter={(e) => {
            if (canGoPrevious && !isLoading) {
              e.currentTarget.style.backgroundColor = '#5a6268';
            }
          }}
          onMouseLeave={(e) => {
            if (canGoPrevious && !isLoading) {
              e.currentTarget.style.backgroundColor = '#6c757d';
            }
          }}
        >
          ← Previous
        </button>

        {/* Image Selector Dropdown */}
        <select
          value={currentImageIndex}
          onChange={handleImageSelect}
          disabled={isLoading}
          style={{
            padding: '0.75rem 1rem',
            borderRadius: '6px',
            border: '1px solid #ddd',
            backgroundColor: '#fff',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            fontSize: '0.9rem',
            minWidth: '200px'
          }}
        >
          {images.map((image, index) => (
            <option key={image.id} value={index}>
              {index + 1}. {image.filename} {image.isCompleted ? '✓' : ''}
            </option>
          ))}
        </select>

        {/* Next Button */}
        <button
          onClick={navigateNext}
          disabled={!canGoNext || isLoading}
          style={{
            padding: '0.75rem 1.5rem',
            backgroundColor: canGoNext && !isLoading ? '#007bff' : '#e9ecef',
            color: canGoNext && !isLoading ? '#fff' : '#6c757d',
            border: 'none',
            borderRadius: '6px',
            cursor: canGoNext && !isLoading ? 'pointer' : 'not-allowed',
            fontSize: '1rem',
            fontWeight: 'bold',
            transition: 'all 0.2s ease',
            minWidth: '120px'
          }}
          onMouseEnter={(e) => {
            if (canGoNext && !isLoading) {
              e.currentTarget.style.backgroundColor = '#0056b3';
            }
          }}
          onMouseLeave={(e) => {
            if (canGoNext && !isLoading) {
              e.currentTarget.style.backgroundColor = '#007bff';
            }
          }}
        >
          Next →
        </button>
      </div>

      {/* Keyboard Shortcuts Info */}
      <div style={{
        marginTop: '1rem',
        padding: '0.5rem',
        fontSize: '0.8rem',
        color: '#666',
        textAlign: 'center',
        borderTop: '1px solid #eee'
      }}>
        <strong>Keyboard shortcuts:</strong> ← → arrow keys to navigate, Space for next image
      </div>

      {/* Loading Indicator */}
      {isLoading && (
        <div style={{
          marginTop: '0.5rem',
          textAlign: 'center',
          color: '#007bff',
          fontSize: '0.9rem'
        }}>
          Loading...
        </div>
      )}
    </div>
  );
};

export default NavigationControls; 