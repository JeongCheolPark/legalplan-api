import time
import subprocess
import os
import traceback
from langchain_ollama import ChatOllama
from app.core.config import settings
import logging

# 로거 설정
logger = logging.getLogger(__name__)

# Tokenizers 병렬 처리 경고 끄기
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_ollama_model():
    """Ollama 모델 설정 및 테스트"""
    logger.info(f"Ollama {settings.OLLAMA_MODEL} 모델 초기화 중...")
    
    try:
        # Ollama 모델 초기화
        start_time = time.time()
        llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.1)
        
        # 간단한 테스트 메시지 보내기
        test_message = "안녕하세요"
        logger.info(f"테스트 메시지 전송: '{test_message}'")
        
        response = llm.invoke(test_message)
        end_time = time.time()
        
        logger.info(f"테스트 응답: {response.content[:100]}...")
        logger.info(f"Ollama 모델 초기화 완료 (소요 시간: {end_time - start_time:.2f}초)")
        
        return llm
    except Exception as e:
        logger.error(f"Ollama 모델 초기화 실패: {str(e)}")
        logger.error(f"상세 오류: {traceback.format_exc()}")
        return None

# 모듈 레벨 인스턴스
llm_instance = None

def init_llm():
    """LLM 모델을 초기화하고 전역 변수로 설정"""
    global llm_instance
    if llm_instance is None:
        llm_instance = setup_ollama_model()
    return llm_instance

def get_llm_instance():
    """현재 LLM 인스턴스 반환 또는 초기화"""
    global llm_instance
    if llm_instance is None:
        return init_llm()
    return llm_instance

def get_streaming_llm():
    """스트리밍용 LLM 인스턴스 반환"""
    logger.info("스트리밉 LLM 초기화...")
    
    try:
        # 스트리밍 모드 LLM 생성
        streaming_llm = ChatOllama(
            model=settings.OLLAMA_MODEL, 
            temperature=0.1,
            streaming=True  # 스트리밍 활성화
        )
        
        logger.info("스트리밍 LLM 초기화 완료")
        return streaming_llm
    except Exception as e:
        logger.error(f"스트리밍 LLM 초기화 실패: {str(e)}")
        return None