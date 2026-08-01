import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Potato Plant Disease Detection API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = ""
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model", "potato_classifier_final.h5")
    )
    
    GREEN_RATIO_THRESH: float = 0.05
    CLASS_NAMES: List[str] = ["early_blight", "late_blight", "healthy"]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
