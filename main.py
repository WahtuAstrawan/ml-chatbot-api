import logging
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from config.settings import get_settings
from models.schemas import ChatRequest, ChatResponse, ContextEntry, ErrorResponse
from services.gemini import get_gemini_service
from services.retrieval import get_retrieval_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def get_app_info():
    """Root endpoint for basic application information."""
    return {
        "name": settings.APP_NAME,
        "description": settings.APP_DESCRIPTION,
    }


@app.post("/chat", response_model=ChatResponse, responses={400: {"model": ErrorResponse}}, tags=["Chat"])
async def chat_with_kakawin_ramayana(
        request: ChatRequest,
        gemini_service=Depends(get_gemini_service),
        retrieval_service=Depends(get_retrieval_service)
):
    """
    Process a chat request and generate a response based on Kakawin Ramayana dataset.

    Args:
        request: The chat request containing query and retrieval parameters

    Returns:
        ChatResponse with generated response and context information
    """
    try:
        # Validate request
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Received chat request: {request.query[:50]}{'...' if len(request.query) > 50 else ''}")

        # Enhance query for better retrieval
        enhanced_query = gemini_service.enhance_query(request.query)
        logger.debug(f"Enhanced query: {enhanced_query}")

        # Retrieve relevant contexts
        contexts = retrieval_service.retrieve(
            enhanced_query,
            top_k=request.top_k,
            context_size=request.context_size
        )

        # Generate response using Gemini
        response_text = gemini_service.generate_response(request.query, contexts)

        # Check if response indicates irrelevant question
        is_irrelevant = "pertanyaan anda tidak relevan" in response_text.strip().lower()

        # Format context entries for response
        context_entries = [] if is_irrelevant else [
            ContextEntry(
                sargah_number=c["sargah_number"],
                sargah_name=c["sargah_name"],
                bait=c["bait"],
                sanskrit_text=c["sanskrit_text"],
                text=c["text"],
                is_top_k=c["is_top_k"]
            )
            for c in contexts
        ]

        return ChatResponse(
            response=response_text,
            context=context_entries
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


@app.get("/sound", tags=["Sound"])
async def get_bait_sound(
    sargah: int = Query(..., gt=0, description="Sargah number (positive integer)"),
    bait: int = Query(..., gt=0, description="Bait number (positive integer)")
):
    """
    Retrieve an MP3 file for the specified sargah and bait.

    Args:
        sargah: The sargah number (required, positive integer)
        bait: The bait number (required, positive integer)

    Returns:
        MP3 file if found, otherwise raises 404 error
    """
    try:
        # Define the path to the sounds folder
        sounds_dir = os.path.join(os.path.dirname(__file__), "sounds")
        file_name = f"{sargah}-{bait}.mp3"
        file_path = os.path.join(sounds_dir, file_name)

        # Check if file exists
        if not os.path.isfile(file_path):
            logger.warning(f"MP3 file not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"MP3 file for sargah {sargah} bait {bait} not found")

        logger.info(f"Serving MP3 file: {file_path}")
        return FileResponse(
            path=file_path,
            media_type="audio/mpeg",
            filename=file_name
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving sound file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sound file: {str(e)}")