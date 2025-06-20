import React from 'react';
import Layout from '../components/Layout';
import DemoInterface from '../components/DemoInterface';
import NavigationControls from '../components/NavigationControls';
import { ImageNavigationProvider } from '../contexts/ImageNavigationContext';

const Demo: React.FC = () => {
  return (
    <Layout>
      <ImageNavigationProvider>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          padding: '2rem',
          minHeight: '100vh'
        }}>
          {/* Page Header */}
          <header style={{
            textAlign: 'center',
            marginBottom: '2rem',
            paddingBottom: '1rem',
            borderBottom: '2px solid #e9ecef'
          }}>
            <h1 style={{
              margin: '0 0 1rem 0',
              color: '#2c3e50',
              fontSize: '2.5rem',
              fontWeight: '600'
            }}>
              Nail Segmentation Demo
            </h1>
            <p style={{
              margin: 0,
              color: '#6c757d',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              Experience real-time nail segmentation powered by our U-Net model. 
              Navigate through images and see instant segmentation results.
            </p>
          </header>

          {/* Navigation Controls */}
          <div style={{
            marginBottom: '2rem',
            padding: '1rem',
            backgroundColor: '#fff',
            borderRadius: '8px',
            border: '1px solid #dee2e6',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <NavigationControls />
          </div>

          {/* Demo Interface */}
          <main>
            <DemoInterface />
          </main>

          {/* Footer Information */}
          <footer style={{
            marginTop: '3rem',
            padding: '2rem',
            backgroundColor: '#f8f9fa',
            borderRadius: '8px',
            border: '1px solid #dee2e6',
            textAlign: 'center'
          }}>
            <h3 style={{
              margin: '0 0 1rem 0',
              color: '#495057',
              fontSize: '1.3rem'
            }}>
              About the Model
            </h3>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '1.5rem',
              textAlign: 'left'
            }}>
              <div>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#6c757d' }}>Architecture</h4>
                <p style={{ margin: 0, color: '#6c757d', fontSize: '0.9rem' }}>
                  U-Net convolutional neural network optimized for precise nail segmentation 
                  with high accuracy and fast inference times.
                </p>
              </div>
              <div>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#6c757d' }}>Performance</h4>
                <p style={{ margin: 0, color: '#6c757d', fontSize: '0.9rem' }}>
                  Real-time processing with sub-second inference times, delivering 
                  high-quality segmentation masks for immediate visualization.
                </p>
              </div>
              <div>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#6c757d' }}>Features</h4>
                <p style={{ margin: 0, color: '#6c757d', fontSize: '0.9rem' }}>
                  Supports various image formats, adjustable thresholds, and confidence scoring 
                  for robust nail detection across different lighting conditions.
                </p>
              </div>
            </div>
          </footer>
        </div>
      </ImageNavigationProvider>
    </Layout>
  );
};

export default Demo;