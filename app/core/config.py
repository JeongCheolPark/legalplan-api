from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings(BaseSettings):
    # 앱 기본 설정
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LegalPlan RAG API"
    
    # CORS 설정
    CORS_ORIGINS: List[str] = ["*"]
    
    # 모델 설정
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:14b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nlpai-lab/KoE5")
    DEVICE: str = os.getenv("DEVICE", "cpu") # "cuda", "mps", "cpu"
    
    # Qdrant 설정
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_db")
    COLLECTION_NAMES: List[str] = ["civil_law", "commercial_law", "criminal_law"]
    
    # 데이터 경로 설정
    DATA_ROOT: str = os.getenv("DATA_ROOT", "./data")
    CIVIL_LAW_DIR: str = f"{DATA_ROOT}/civil_law"
    COMMERCIAL_LAW_DIR: str = f"{DATA_ROOT}/commercial_law"
    CRIMINAL_LAW_DIR: str = f"{DATA_ROOT}/criminal_law"
    
    # 청크 설정
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Supabase 설정
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()