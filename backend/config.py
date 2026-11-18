from pathlib import Path
import os
from typing import List
from dotenv import load_dotenv

# Resolve repository root and backend directories
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent

# Load .env file from root or backend directory if present
env_path = ROOT_DIR / ".env"
if not env_path.exists():
    env_path = BACKEND_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


class Settings:
    PROJECT_NAME: str = "TruthLens AI"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "AI-Powered Fake News Detection and Evidence-Based Fact Verification API"
    
    # Base paths
    ROOT_PATH: Path = ROOT_DIR
    BACKEND_PATH: Path = BACKEND_DIR
    DATA_PATH: Path = ROOT_DIR / "data"
    MODELS_PATH: Path = ROOT_DIR / "models"
    EVALUATION_PATH: Path = ROOT_DIR / "evaluation"

    # Model paths
    _distilbert_env = os.getenv("DISTILBERT_MODEL_PATH", "models/distilbert_final_256/checkpoint-7821")
    DISTILBERT_MODEL_PATH: Path = (ROOT_DIR / _distilbert_env).resolve() if not Path(_distilbert_env).is_absolute() else Path(_distilbert_env)

    _xgboost_env = os.getenv("XGBOOST_MODEL_PATH", "models/xgboost_model.joblib")
    XGBOOST_MODEL_PATH: Path = (ROOT_DIR / _xgboost_env).resolve() if not Path(_xgboost_env).is_absolute() else Path(_xgboost_env)

    _tfidf_env = os.getenv("TFIDF_VECTORIZER_PATH", "models/tfidf_vectorizer.joblib")
    TFIDF_VECTORIZER_PATH: Path = (ROOT_DIR / _tfidf_env).resolve() if not Path(_tfidf_env).is_absolute() else Path(_tfidf_env)

    # Server settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS Origins
    _raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000")
    CORS_ORIGINS: List[str] = [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]

    # LLM settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "offline").lower()
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Retrieval settings
    RETRIEVAL_PROVIDER: str = os.getenv("RETRIEVAL_PROVIDER", "duckduckgo").lower()
    MAX_RETRIEVED_DOCS: int = int(os.getenv("MAX_RETRIEVED_DOCS", "5"))
    RETRIEVAL_TIMEOUT_SECONDS: int = int(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "10"))


settings = Settings()
