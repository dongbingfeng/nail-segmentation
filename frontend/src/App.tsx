import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import About from './pages/About';
import Labeling from './pages/Labeling';
import Demo from './pages/Demo';
import './App.css';

const App: React.FC = () => (
  <Router>
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/labeling" element={<Labeling />} />
        <Route path="/demo" element={<Demo />} />
      </Routes>
    </Layout>
  </Router>
);

export default App; 