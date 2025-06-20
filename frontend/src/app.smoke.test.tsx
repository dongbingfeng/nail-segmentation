import React from 'react';
import { render } from '@testing-library/react';
import Home from './pages/Home';
import About from './pages/About';

describe('Components', () => {
  it('renders Home component', () => {
    const { getByText } = render(<Home />);
    expect(getByText(/Welcome to Nail Segmentation/i)).toBeInTheDocument();
  });

  it('renders About component', () => {
    const { getByText } = render(<About />);
    expect(getByText(/About/i)).toBeInTheDocument();
    expect(getByText(/web-based platform/i)).toBeInTheDocument();
  });
}); 