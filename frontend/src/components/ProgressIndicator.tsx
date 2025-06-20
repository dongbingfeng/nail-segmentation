import React from 'react';
import { useImageNavigation } from '../contexts/ImageNavigationContext';

interface ProgressIndicatorProps {
  className?: string;
  showDetails?: boolean;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ 
  className = '', 
  showDetails = true 
}) => {
  const { progress, images, currentImage } = useImageNavigation();

  if (!progress || images.length === 0) {
    return (
      <div className={`progress-indicator ${className}`} style={{
        padding: '1rem',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px',
        border: '1px solid #ddd',
        textAlign: 'center'
      }}>
        <p style={{ margin: 0, color: '#666' }}>No progress data available</p>
      </div>
    );
  }

  const progressPercentage = Math.max(0, Math.min(100, progress.percentComplete));
  const completedImages = progress.completedImages;
  const totalImages = progress.totalImages;

  // Calculate progress bar color based on completion
  const getProgressColor = (percentage: number) => {
    if (percentage === 0) return '#e9ecef';
    if (percentage < 25) return '#dc3545';
    if (percentage < 50) return '#fd7e14';
    if (percentage < 75) return '#ffc107';
    if (percentage < 100) return '#28a745';
    return '#28a745';
  };

  const progressColor = getProgressColor(progressPercentage);

  return (
    <div className={`progress-indicator ${className}`} style={{
      padding: '1rem',
      backgroundColor: '#fff',
      borderRadius: '8px',
      border: '1px solid #ddd',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '1rem'
      }}>
        <h3 style={{ 
          margin: 0, 
          fontSize: '1.1rem', 
          color: '#333' 
        }}>
          Labeling Progress
        </h3>
        <div style={{ 
          fontSize: '1.2rem', 
          fontWeight: 'bold', 
          color: progressColor 
        }}>
          {progressPercentage.toFixed(1)}%
        </div>
      </div>

      {/* Progress Bar */}
      <div style={{
        width: '100%',
        height: '20px',
        backgroundColor: '#e9ecef',
        borderRadius: '10px',
        overflow: 'hidden',
        marginBottom: '1rem',
        position: 'relative'
      }}>
        <div style={{
          width: `${progressPercentage}%`,
          height: '100%',
          backgroundColor: progressColor,
          borderRadius: '10px',
          transition: 'width 0.3s ease, background-color 0.3s ease',
          position: 'relative'
        }}>
          {/* Progress bar shine effect */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '50%',
            background: 'linear-gradient(to bottom, rgba(255,255,255,0.3), transparent)',
            borderRadius: '10px 10px 0 0'
          }} />
        </div>
      </div>

      {/* Statistics */}
      {showDetails && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
          gap: '1rem',
          marginBottom: '1rem'
        }}>
          <div style={{
            textAlign: 'center',
            padding: '0.5rem',
            backgroundColor: '#f8f9fa',
            borderRadius: '6px'
          }}>
            <div style={{ 
              fontSize: '1.5rem', 
              fontWeight: 'bold', 
              color: '#28a745' 
            }}>
              {completedImages}
            </div>
            <div style={{ 
              fontSize: '0.8rem', 
              color: '#666' 
            }}>
              Completed
            </div>
          </div>
        
          <div style={{
            textAlign: 'center',
            padding: '0.5rem',
            backgroundColor: '#f8f9fa',
            borderRadius: '6px'
          }}>
            <div style={{ 
              fontSize: '1.5rem', 
              fontWeight: 'bold', 
              color: '#007bff' 
            }}>
              {totalImages}
            </div>
            <div style={{ 
              fontSize: '0.8rem', 
              color: '#666' 
            }}>
              Total
            </div>
          </div>
        </div>
      )}

      {/* Current Image Status */}
      {/* currentImage && (
        <div style={{
          padding: '0.75rem',
          backgroundColor: currentImage.isCompleted ? '#d4edda' : '#fff3cd',
          border: `1px solid ${currentImage.isCompleted ? '#c3e6cb' : '#ffeaa7'}`,
          borderRadius: '6px',
          fontSize: '0.9rem'
        }}>
          <div style={{ 
            fontWeight: 'bold', 
            marginBottom: '0.25rem',
            color: currentImage.isCompleted ? '#155724' : '#856404'
          }}>
            Current: {currentImage.filename}
          </div>
          <div style={{ 
            color: currentImage.isCompleted ? '#155724' : '#856404' 
          }}>
            Status: {currentImage.isCompleted ? '✅ Completed' : '⏳ In Progress'} 
            {currentImage.annotationCount > 0 && (
              <span style={{ marginLeft: '0.5rem' }}>
                ({currentImage.annotationCount} annotations)
              </span>
            )}
          </div>
        </div>
      ) */}

      {/* Completion Message */}
      {/* progressPercentage === 100 && (
        <div style={{
          marginTop: '1rem',
          padding: '1rem',
          backgroundColor: '#d4edda',
          border: '1px solid #c3e6cb',
          borderRadius: '6px',
          textAlign: 'center',
          color: '#155724'
        }}>
          <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
            🎉 All images labeled!
          </div>
          <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
            Great job! You've completed labeling all {totalImages} images.
          </div>
        </div>
      ) */}
    </div>
  );
};

export default ProgressIndicator; 