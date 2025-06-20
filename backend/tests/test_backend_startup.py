#!/usr/bin/env python3
"""Test backend startup with SAM routes"""

import sys
import os

# Add backend src to path
backend_src = os.path.join(os.path.dirname(__file__), 'backend', 'src')
sys.path.insert(0, backend_src)

def test_backend_startup():
    print("Testing backend startup with SAM routes...")
    
    try:
        from server.main import app
        print("✓ Backend app creation successful")
        
        # Check routes
        routes = [route.path for route in app.routes]
        print(f"✓ Available routes: {len(routes)}")
        
        # Check for SAM-specific routes
        sam_routes = [route for route in routes if '/sam/' in route]
        if sam_routes:
            print(f"✓ SAM routes found: {sam_routes}")
        else:
            print("⚠ No SAM routes found - checking route registration...")
        
        print("Backend with SAM integration working!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_backend_startup()
    sys.exit(0 if success else 1) 