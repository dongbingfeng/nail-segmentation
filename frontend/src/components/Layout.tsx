import React from 'react';
import Navbar from './Navbar';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => (
  <div className="layout">
    <Navbar />
    <main>{children}</main>
    <footer style={{ textAlign: 'center', padding: '1rem', background: '#f5f5f5' }}>
      &copy; {new Date().getFullYear()} Nail Segmentation
    </footer>
  </div>
);

export default Layout; 