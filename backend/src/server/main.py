from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List
import logging
import asyncio

from server.api_routes import router as api_router
from server.sam_routes import router as sam_router
from server.unet_routes import router as unet_router, initialize_unet_service

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting up services...")
    
    # Initialize U-Net service
    try:
        await initialize_unet_service()
        logger.info("U-Net service initialized successfully")
    except Exception as e:
        logger.warning(f"U-Net service initialization failed: {e}")
        logger.warning("U-Net endpoints will return 503 until manually initialized")
    
    logger.info("Startup complete")
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down services...")


app = FastAPI(
    title="Nail Segmentation Backend", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/")
def read_root():
    return {"message": "Nail Segmentation Backend is running."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Basic test endpoints (CRUD functionality removed to focus on image labeling)
@app.get("/test")
def test_endpoint():
    return {"message": "Backend is working correctly", "endpoints": ["/api/images", "/api/labeling/progress"]}

# Include image labeling API routes
app.include_router(api_router)

# Include SAM API routes  
app.include_router(sam_router)

# Include U-Net API routes
app.include_router(unet_router)