import React from 'react';

interface ToolPaletteProps {
  selectedTool: string;
  onToolSelect: (tool: string) => void;
  onSamProcess: () => void;
  onClearSamPoints: () => void;
  onClearMaskPoints: () => void;
  onSaveLabel: () => void;
  isProcessing: boolean;
  samPointsCount: number;
  maskPointsCount: number;
  boundingBoxCount: number;
  totalAnnotationCount: number;
}

const ToolPalette: React.FC<ToolPaletteProps> = ({
  selectedTool,
  onToolSelect,
  onSamProcess,
  onClearSamPoints,
  onClearMaskPoints,
  onSaveLabel,
  isProcessing,
  samPointsCount,
  maskPointsCount,
  boundingBoxCount,
  totalAnnotationCount
}) => {
  const tools = [
    { id: 'select', name: 'Select', icon: '🔍' },
    { id: 'bounding_box', name: 'Bounding Box', icon: '📦' },
    { id: 'sam_point', name: 'SAM Point', icon: '📍' },
  ];

  return (
    <div style={{
      padding: '1rem',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      marginBottom: '1rem'
    }}>
      <h3 style={{ 
        margin: '0 0 1rem 0', 
        fontSize: '1.1rem', 
        color: '#333' 
      }}>
        Labeling Tools
      </h3>
      
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
        {tools.map((tool) => (
          <button
            key={tool.id}
            onClick={() => onToolSelect(tool.id)}
            style={{
              padding: '0.75rem 1rem',
              border: selectedTool === tool.id ? '2px solid #1976d2' : '2px solid #ddd',
              borderRadius: '6px',
              backgroundColor: selectedTool === tool.id ? '#e3f2fd' : '#fff',
              color: selectedTool === tool.id ? '#1976d2' : '#666',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '0.25rem',
              fontSize: '0.9rem',
              fontWeight: selectedTool === tool.id ? 'bold' : 'normal',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              if (selectedTool !== tool.id) {
                e.currentTarget.style.backgroundColor = '#f5f5f5';
                e.currentTarget.style.borderColor = '#bbb';
              }
            }}
            onMouseLeave={(e) => {
              if (selectedTool !== tool.id) {
                e.currentTarget.style.backgroundColor = '#fff';
                e.currentTarget.style.borderColor = '#ddd';
              }
            }}
          >
            <span style={{ fontSize: '1.2rem' }}>{tool.icon}</span>
            <span>{tool.name}</span>
          </button>
        ))}
      </div>

      <div style={{ borderTop: '1px solid #ddd', paddingTop: '1rem' }}>
        {/* Bounding Box Status */}
        {boundingBoxCount > 0 && (
          <div style={{
            marginBottom: '0.75rem',
            padding: '0.5rem',
            backgroundColor: '#fff3e0',
            borderRadius: '4px',
            fontSize: '0.9rem',
            color: '#f57c00',
            textAlign: 'center'
          }}>
            Bounding Box: {boundingBoxCount}
          </div>
        )}

        {/* SAM Points Status */}
        {samPointsCount > 0 && (
          <div style={{
            marginBottom: '0.75rem',
            padding: '0.5rem',
            backgroundColor: '#e8f5e8',
            borderRadius: '4px',
            fontSize: '0.9rem',
            color: '#2e7d32',
            textAlign: 'center'
          }}>
            SAM Points: {samPointsCount}
          </div>
        )}

        {/* Mask Points Status */}
        {maskPointsCount > 0 && (
          <div style={{
            marginBottom: '0.75rem',
            padding: '0.5rem',
            backgroundColor: '#e8f4fd',
            borderRadius: '4px',
            fontSize: '0.9rem',
            color: '#1976d2',
            textAlign: 'center'
          }}>
            Mask Points: {maskPointsCount}
          </div>
        )}

        {/* SAM Process Button */}
        <button
          onClick={onSamProcess}
          disabled={isProcessing || boundingBoxCount === 0}
          style={{
            width: '100%',
            padding: '0.75rem',
            border: 'none',
            borderRadius: '6px',
            backgroundColor: isProcessing || boundingBoxCount === 0 ? '#ccc' : '#4caf50',
            color: '#fff',
            cursor: isProcessing || boundingBoxCount === 0 ? 'not-allowed' : 'pointer',
            fontSize: '1rem',
            fontWeight: 'bold',
            transition: 'background-color 0.2s ease',
            marginBottom: '0.5rem'
          }}
          onMouseEnter={(e) => {
            if (!isProcessing && boundingBoxCount > 0) {
              e.currentTarget.style.backgroundColor = '#45a049';
            }
          }}
          onMouseLeave={(e) => {
            if (!isProcessing && boundingBoxCount > 0) {
              e.currentTarget.style.backgroundColor = '#4caf50';
            }
          }}
        >
          {isProcessing 
            ? 'Processing SAM...' 
            : samPointsCount > 0 
              ? `Process with SAM (${samPointsCount} points)` 
              : 'Process with SAM (full region)'
          }
        </button>

        {/* Save All Button - Save All Annotations */}
        {(maskPointsCount > 0 || totalAnnotationCount > 0) && (
          <button
            onClick={onSaveLabel}
            disabled={isProcessing}
            style={{
              width: '100%',
              padding: '0.75rem',
              border: 'none',
              borderRadius: '6px',
              backgroundColor: isProcessing ? '#ccc' : '#4caf50',
              color: '#fff',
              cursor: isProcessing ? 'not-allowed' : 'pointer',
              fontSize: '1rem',
              fontWeight: 'bold',
              transition: 'background-color 0.2s ease',
              marginBottom: '0.5rem'
            }}
            onMouseEnter={(e) => {
              if (!isProcessing) {
                e.currentTarget.style.backgroundColor = '#45a049';
              }
            }}
            onMouseLeave={(e) => {
              if (!isProcessing) {
                e.currentTarget.style.backgroundColor = '#4caf50';
              }
            }}
          >
            {maskPointsCount > 0 
              ? `Save All (${totalAnnotationCount} annotations + ${maskPointsCount} mask points)`
              : `Save All (${totalAnnotationCount} annotations)`
            }
          </button>
        )}

        {/* Relabel Button - Clear Mask Points */}
        {maskPointsCount > 0 && (
          <button
            onClick={onClearMaskPoints}
            disabled={isProcessing}
            style={{
              width: '100%',
              padding: '0.75rem',
              border: 'none',
              borderRadius: '6px',
              backgroundColor: isProcessing ? '#ccc' : '#1976d2',
              color: '#fff',
              cursor: isProcessing ? 'not-allowed' : 'pointer',
              fontSize: '1rem',
              fontWeight: 'bold',
              transition: 'background-color 0.2s ease',
              marginBottom: '0.5rem'
            }}
            onMouseEnter={(e) => {
              if (!isProcessing) {
                e.currentTarget.style.backgroundColor = '#1565c0';
              }
            }}
            onMouseLeave={(e) => {
              if (!isProcessing) {
                e.currentTarget.style.backgroundColor = '#1976d2';
              }
            }}
          >
            Relabel (Clear Mask Points)
          </button>
        )}

        {/* Clear SAM Points Button */}
        {samPointsCount > 0 && (
          <button
            onClick={onClearSamPoints}
            disabled={isProcessing}
            style={{
              width: '100%',
              padding: '0.5rem',
              border: '1px solid #ff9800',
              borderRadius: '4px',
              backgroundColor: '#fff',
              color: '#ff9800',
              cursor: isProcessing ? 'not-allowed' : 'pointer',
              fontSize: '0.9rem',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => {
              if (!isProcessing) {
                e.currentTarget.style.backgroundColor = '#fff3e0';
              }
            }}
            onMouseLeave={(e) => {
              if (!isProcessing) {
                e.currentTarget.style.backgroundColor = '#fff';
              }
            }}
          >
            Clear SAM Points
          </button>
        )}
      </div>

      <div style={{ 
        marginTop: '1rem', 
        fontSize: '0.8rem', 
        color: '#666',
        lineHeight: '1.4'
      }}>
        <p style={{ margin: '0.5rem 0' }}>
          <strong>Select:</strong> Move and resize existing annotations
        </p>
        <p style={{ margin: '0.5rem 0' }}>
          <strong>Bounding Box:</strong> Draw rectangular annotations
        </p>
        <p style={{ margin: '0.5rem 0' }}>
          <strong>SAM Point:</strong> Click to generate segmentation
        </p>
      </div>
    </div>
  );
};

export default ToolPalette; 