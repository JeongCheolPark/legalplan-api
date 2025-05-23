import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.config import settings
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def get_embeddings():
    """KoE5 임베딩 모델을 초기화하고 반환"""
    logger.info("KoE5 임베딩 모델 초기화 중...")
    
    start_time = time.time()
    
    # KoE5 모델 로드
    model_kwargs = {'device': settings.DEVICE}
    encode_kwargs = {'normalize_embeddings': True}
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # 간단한 테스트로 임베딩 생성
        test_text = "법률 자문 테스트"
        test_embedding = embeddings.embed_query(test_text)
        
        end_time = time.time()
        
        logger.info(f"임베딩 차원: {len(test_embedding)}")
        logger.info(f"KoE5 임베딩 모델 초기화 완료 (소요 시간: {end_time - start_time:.2f}초)")
        
        return embeddings
    except Exception as e:
        logger.error(f"임베딩 모델 초기화 실패: {str(e)}")
        raise

# 모듈 레벨에서 미리 임베딩 인스턴스 생성
embeddings_model = None

def init_embeddings():
    """임베딩 모델을 초기화하고 전역 변수로 설정"""
    global embeddings_model
    if embeddings_model is None:
        embeddings_model = get_embeddings()
    return embeddings_model

def get_embeddings_instance():
    """현재 임베딩 모델 인스턴스 반환 또는 초기화"""
    global embeddings_model
    if embeddings_model is None:
        return init_embeddings()
    return embeddings_model