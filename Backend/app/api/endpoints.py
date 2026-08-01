import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.schemas.prediction import PredictionResponse, ErrorResponse
from app.services.image_service import leaf_present_bgr, preprocess_image
from app.services.model_service import model_service
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or non-image format"},
        500: {"model": ErrorResponse, "description": "Internal server / prediction error"}
    }
)
async def predict_disease(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Uploaded file content-type '{file.content_type}' is not image")
    
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File could not be decoded as a valid image"
            )
        
        # Check for leaf presence based on green pixel threshold
        green_ratio = leaf_present_bgr(img)
        logger.info(f"Image processed. Calculated green_ratio={green_ratio:.4f}")
        
        if green_ratio < settings.GREEN_RATIO_THRESH:
            logger.info("Green ratio below threshold. Returning 'no_leaf'.")
            return PredictionResponse(
                label="no_leaf",
                confidence=1.0,
                message="No plant leaf detected in the image"
            )
        
        # Preprocess and predict
        input_tensor = preprocess_image(img)
        label, confidence = model_service.predict(input_tensor)
        
        logger.info(f"Prediction result: label={label}, confidence={confidence:.4f}")
        return PredictionResponse(label=label, confidence=confidence)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
