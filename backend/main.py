from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import logging
import asyncio
import argparse
from src.server.api_routes import router as api_router
from src.server.image_service import ImageService

# Global configuration for enabled services
ENABLED_SERVICES = {
    'sam': True,
    'unet': True
}

app = FastAPI(title="Nail Segmentation API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/")
async def root():
    return {"message": "Nail Segmentation API is running!"}

@app.get("/health")
async def health_check():
    """Comprehensive health check for all enabled services"""
    health_status = {
        "status": "healthy",
        "services": {},
        "enabled_services": ENABLED_SERVICES,
        "timestamp": None
    }
    
    try:
        # Check image service
        image_service = ImageService()
        images = await image_service.get_image_list()
        health_status["services"]["image_service"] = {
            "status": "healthy",
            "images_found": len(images),
            "message": f"Found {len(images)} images for labeling"
        }
    except Exception as e:
        health_status["services"]["image_service"] = {
            "status": "error", 
            "message": f"Image service failed: {str(e)}"
        }
        health_status["status"] = "degraded"
    
    # Check SAM service health (only if enabled)
    if ENABLED_SERVICES['sam']:
        try:
            from src.server.sam_routes import get_sam_service
            sam_service = await get_sam_service()
            sam_health = await sam_service.health_check()
            health_status["services"]["sam_service"] = sam_health
            
            if not sam_health.get("model_loaded", False):
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["sam_service"] = {
                "status": "error",
                "message": f"SAM service failed: {str(e)}",
                "model_loaded": False
            }
            health_status["status"] = "degraded"
    else:
        health_status["services"]["sam_service"] = {
            "status": "disabled",
            "message": "SAM service not enabled"
        }
    
    # Check UNet service health (only if enabled)
    if ENABLED_SERVICES['unet']:
        try:
            from src.server.unet_routes import get_unet_service
            unet_service = get_unet_service()
            unet_health = await unet_service.get_health_status()
            health_status["services"]["unet_service"] = unet_health
            
            if not unet_health.get("model_loaded", False):
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["unet_service"] = {
                "status": "error",
                "message": f"UNet service failed: {str(e)}",
                "model_loaded": False
            }
            health_status["status"] = "degraded"
    else:
        health_status["services"]["unet_service"] = {
            "status": "disabled",
            "message": "UNet service not enabled"
        }
    
    import datetime
    health_status["timestamp"] = datetime.datetime.utcnow().isoformat()
    
    return health_status

@app.get("/test")
def test_endpoint():
    """Test endpoint showing all available API routes"""
    endpoints = {
        "image_service": [
            "/api/images",
            "/api/images/{image_id}/annotations", 
            "/api/labeling/progress"
        ],
        "health": [
            "/health"
        ]
    }
    
    if ENABLED_SERVICES['sam']:
        endpoints["sam_service"] = [
            "/api/sam/segment",
            "/api/sam/health",
            "/api/sam/model-info",
            "/api/sam/initialize"
        ]
    
    if ENABLED_SERVICES['unet']:
        endpoints["unet_service"] = [
            "/api/unet/segment",
            "/api/unet/segment-batch",
            "/api/unet/health",
            "/api/unet/model-info",
            "/api/unet/initialize"
        ]
    
    return {
        "message": "Backend is working correctly", 
        "enabled_services": ENABLED_SERVICES,
        "endpoints": endpoints
    }

# Include image labeling API routes
app.include_router(api_router, prefix="/api")

# Setup service routers immediately at startup
def setup_service_routers():
    """Setup and include service routers based on enabled services."""
    # Conditionally include service routers based on enabled services
    if ENABLED_SERVICES['sam']:
        from src.server.sam_routes import router as sam_router
        app.include_router(sam_router)

    if ENABLED_SERVICES['unet']:
        from src.server.unet_routes import router as unet_router
        app.include_router(unet_router)

# Call setup immediately at module level
setup_service_routers()

def parse_arguments():
    """Parse command line arguments for service selection."""
    parser = argparse.ArgumentParser(
        description="Nail Segmentation API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Service Selection Examples:
  python main.py --service sam          # Start only SAM service
  python main.py --service unet         # Start only UNet service  
  python main.py --service both         # Start both services (default)
  python main.py --sam-only             # Alternative: SAM only
  python main.py --unet-only            # Alternative: UNet only
        """
    )
    
    # Service selection options
    service_group = parser.add_mutually_exclusive_group()
    service_group.add_argument(
        '--service',
        choices=['sam', 'unet', 'both'],
        default='both',
        help='Choose which service(s) to start (default: both)'
    )
    service_group.add_argument(
        '--sam-only',
        action='store_true',
        help='Start only the SAM service'
    )
    service_group.add_argument(
        '--unet-only', 
        action='store_true',
        help='Start only the UNet service'
    )
    
    # Server configuration
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the server to (default: 8000)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    return parser.parse_args()

def configure_services(args):
    """Configure which services to enable based on command line arguments."""
    global ENABLED_SERVICES
    
    if args.sam_only:
        ENABLED_SERVICES['sam'] = True
        ENABLED_SERVICES['unet'] = False
    elif args.unet_only:
        ENABLED_SERVICES['sam'] = False
        ENABLED_SERVICES['unet'] = True
    elif args.service == 'sam':
        ENABLED_SERVICES['sam'] = True
        ENABLED_SERVICES['unet'] = False
    elif args.service == 'unet':
        ENABLED_SERVICES['sam'] = False
        ENABLED_SERVICES['unet'] = True
    elif args.service == 'both':
        ENABLED_SERVICES['sam'] = True
        ENABLED_SERVICES['unet'] = True
    
    print(f"🔧 Service Configuration:")
    print(f"   SAM Service: {'✅ Enabled' if ENABLED_SERVICES['sam'] else '❌ Disabled'}")
    print(f"   UNet Service: {'✅ Enabled' if ENABLED_SERVICES['unet'] else '❌ Disabled'}")

async def startup_check():
    """Perform startup checks and initialize enabled services."""
    print("🚀 Starting Nail Segmentation API server...")
    print("📁 Scanning for images...")
    
    # Initialize image service to show startup info
    try:
        image_service = ImageService()
        images = await image_service.get_image_list()
        print(f"✅ Found {len(images)} images ready for labeling")
        
        if images:
            for img in images:
                print(f"  - {img.filename} ({img.annotationCount} annotations)")
        else:
            print("  ⚠️  No images found in frontend/public/images/ directory")
    except Exception as e:
        print(f"  ❌ Image service initialization failed: {str(e)}")
    
    # Initialize SAM service (only if enabled)
    if ENABLED_SERVICES['sam']:
        print("\n🤖 Initializing SAM service...")
        try:
            from src.server.sam_routes import get_sam_service
            sam_service = await get_sam_service()
            health_info = await sam_service.health_check()
            
            if health_info.get("model_loaded", False):
                print(f"✅ SAM model loaded successfully")
                print(f"  - Model: {health_info.get('model_name', 'Unknown')}")
                print(f"  - Device: {health_info.get('device', 'Unknown')}")
            else:
                print(f"⚠️  SAM model not fully loaded: {health_info.get('status', 'Unknown')}")
                
            success = await sam_service.initialize_model()
            if success:
                print(f"✅ SAM model initialized successfully")
            else:
                print(f"⚠️  SAM model initialization failed")
        except Exception as e:
            print(f"  ❌ SAM service initialization failed: {str(e)}")
            print("  ℹ️  SAM will initialize on first request")
    else:
        print("\n🤖 SAM service disabled")
    
    # Initialize UNet service (only if enabled)
    if ENABLED_SERVICES['unet']:
        print("\n🧠 Initializing UNet service...")
        try:
            from src.server.unet_routes import initialize_unet_service
            await initialize_unet_service(config_path="./config.yaml")
            print(f"✅ UNet service initialized successfully")
        except Exception as e:
            print(f"  ❌ UNet service initialization failed: {str(e)}")
            print("  ℹ️  UNet will initialize on first request")
    else:
        print("\n🧠 UNet service disabled")

if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    args = parse_arguments()
    configure_services(args)
    
    async def run_startup_and_server():
        """Run startup checks and then start the server."""
        await startup_check()
        
        print(f"\n🌐 Server starting on http://{args.host}:{args.port}")
        print(f"📚 API documentation available at http://{args.host}:{args.port}/docs")
        print(f"🔍 Health check available at http://{args.host}:{args.port}/health")
    
    # Run startup check
    asyncio.run(run_startup_and_server())
    
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)