import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const location = useLocation();
  return (
    <nav style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '1rem',
      background: '#1976d2',
      color: '#fff',
    }}>
      <div style={{ fontWeight: 'bold', fontSize: '1.2rem' }}>Nail Segmentation</div>
      <div>
        <Link
          to="/"
          style={{
            color: location.pathname === '/' ? '#ffd600' : '#fff',
            marginRight: '1rem',
            textDecoration: 'none',
            fontWeight: location.pathname === '/' ? 'bold' : 'normal',
          }}
        >
          Home
        </Link>
        <Link
          to="/labeling"
          style={{
            color: location.pathname === '/labeling' ? '#ffd600' : '#fff',
            marginRight: '1rem',
            textDecoration: 'none',
            fontWeight: location.pathname === '/labeling' ? 'bold' : 'normal',
          }}
        >
          Labeling
        </Link>
        <Link
          to="/demo"
          style={{
            color: location.pathname === '/demo' ? '#ffd600' : '#fff',
            marginRight: '1rem',
            textDecoration: 'none',
            fontWeight: location.pathname === '/demo' ? 'bold' : 'normal',
          }}
        >
          Demo
        </Link>
        <Link
          to="/about"
          style={{
            color: location.pathname === '/about' ? '#ffd600' : '#fff',
            textDecoration: 'none',
            fontWeight: location.pathname === '/about' ? 'bold' : 'normal',
          }}
        >
          About
        </Link>
      </div>
    </nav>
  );
};

export default Navbar; 