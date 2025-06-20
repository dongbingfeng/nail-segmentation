import React from 'react';
import LabelingInterface from '../components/LabelingInterface';
import NavigationControls from '../components/NavigationControls';
import ProgressIndicator from '../components/ProgressIndicator';
import { ImageNavigationProvider } from '../contexts/ImageNavigationContext';

const Labeling: React.FC = () => {
  return (
    <ImageNavigationProvider>
      <div style={{ 
        padding: '1rem',
        minHeight: 'calc(100vh - 80px)', // Account for navbar height
        backgroundColor: '#f5f5f5'
      }}>
        <div style={{
          maxWidth: '1600px',
          margin: '0 auto'
        }}>
          <header style={{
            marginBottom: '2rem',
            textAlign: 'center'
          }}>
            <h1 style={{ 
              color: '#1976d2',
              margin: '0 0 0.5rem 0',
              fontSize: '2rem'
            }}>
              Nail Segmentation Labeling Tool
            </h1>
            <p style={{ 
              color: '#666',
              fontSize: '1.1rem',
              margin: 0
            }}>
              Create and manage annotations for nail segmentation training data
            </p>
          </header>

          {/* Navigation and Progress Section */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 350px',
            gap: '1rem',
            marginBottom: '2rem'
          }}>
            <NavigationControls />
            <ProgressIndicator />
          </div>

          {/* Main Labeling Interface */}
          <div style={{
            backgroundColor: '#fff',
            borderRadius: '12px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            overflow: 'hidden'
          }}>
            <div style={{
              padding: '1rem 1.5rem',
              backgroundColor: '#1976d2',
              color: '#fff',
              borderBottom: '1px solid #1565c0'
            }}>
              <h2 style={{ 
                margin: 0,
                fontSize: '1.2rem',
                fontWeight: '500'
              }}>
                Image Labeling Interface
              </h2>
            </div>

            <div style={{ padding: 0 }}>
              <LabelingInterface />
            </div>
          </div>

          <footer style={{
            marginTop: '2rem',
            padding: '1rem',
            textAlign: 'center',
            color: '#666',
            fontSize: '0.9rem'
          }}>
            <div style={{ marginBottom: '0.5rem' }}>
              <strong>Instructions:</strong>
            </div>
            <div style={{ lineHeight: '1.6' }}>
              • Select the <strong>Bounding Box</strong> tool and drag to create rectangular annotations<br />
              • Use <strong>Select</strong> tool to move and resize existing annotations<br />
              • Use <strong>Next/Previous</strong> buttons or arrow keys to navigate between images<br />
              • Press <strong>Space</strong> to advance to the next image<br />
              • Click <strong>Process with SAM</strong> to generate automatic segmentation<br />
              • View and manage all annotations in the left panel
            </div>
          </footer>
        </div>
      </div>
    </ImageNavigationProvider>
  );
};

export default Labeling; 