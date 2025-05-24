from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import sys

# 설정 모듈 불러오기
from app.core.config import settings

# 로깅 설정 강화
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="법률 RAG 시스템 API",
    version="0.1.0",
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 인스턴스 변수들
embeddings_instance = None
llm_instance = None
vector_stores = None
rag_pipeline_instance = None

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트 핸들러"""
    # 명시적인 출력 추가
    print("==== 애플리케이션 시작 중... ====")
    logger.info("애플리케이션 시작 중...")
    
    # 데이터 디렉토리 확인 및 생성
    for dir_path in [
        settings.DATA_ROOT, 
        settings.CIVIL_LAW_DIR, 
        settings.COMMERCIAL_LAW_DIR, 
        settings.CRIMINAL_LAW_DIR, 
        settings.QDRANT_PATH
    ]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"디렉토리 생성됨: {dir_path}")
            logger.info(f"디렉토리 생성됨: {dir_path}")
        else:
            print(f"디렉토리 확인됨: {dir_path}")
            logger.info(f"디렉토리 확인됨: {dir_path}")
    
    # 환경 설정 정보 출력
    print(f"사용 모델: {settings.OLLAMA_MODEL}")
    logger.info(f"사용 모델: {settings.OLLAMA_MODEL}")
    print(f"임베딩 모델: {settings.EMBEDDING_MODEL}")
    logger.info(f"임베딩 모델: {settings.EMBEDDING_MODEL}")
    print(f"기기: {settings.DEVICE}")
    logger.info(f"기기: {settings.DEVICE}")
    
    # 임베딩 모델 초기화
    try:
        from app.services.embeddings import init_embeddings
        global embeddings_instance
        embeddings_instance = init_embeddings()
        print("임베딩 모델 초기화 완료")
        logger.info("임베딩 모델 초기화 완료")
    except Exception as e:
        print(f"임베딩 모델 초기화 실패: {str(e)}")
        logger.error(f"임베딩 모델 초기화 실패: {str(e)}")
    
    # LLM 초기화
    try:
        from app.services.llm import init_llm
        global llm_instance
        llm_instance = init_llm()
        print("LLM 초기화 완료")
        logger.info("LLM 초기화 완료")
    except Exception as e:
        print(f"LLM 초기화 실패: {str(e)}")
        logger.error(f"LLM 초기화 실패: {str(e)}")
    
    # 벡터 스토어 초기화
    try:
        from app.services.vectorstore import init_vector_stores
        global vector_stores
        vector_stores = init_vector_stores()
        print(f"{len(vector_stores)}개 벡터 스토어 초기화 완료")
        logger.info(f"{len(vector_stores)}개 벡터 스토어 초기화 완료")
    except Exception as e:
        print(f"벡터 스토어 초기화 실패: {str(e)}")
        logger.error(f"벡터 스토어 초기화 실패: {str(e)}")
    
    # RAG 파이프라인 초기화
    try:
        from app.services.rag import init_rag_pipeline
        global rag_pipeline_instance
        rag_pipeline_instance = init_rag_pipeline()
        print("RAG 파이프라인 초기화 완료")
        logger.info("RAG 파이프라인 초기화 완료")
    except Exception as e:
        print(f"RAG 파이프라인 초기화 실패: {str(e)}")
        logger.error(f"RAG 파이프라인 초기화 실패: {str(e)}")
    
    print("==== 애플리케이션 시작 완료 ====")
    logger.info("애플리케이션 시작 완료")

@app.get("/")
async def root():
    logger.info("Root 엔드포인트 호출됨")
    return {"message": "LegalPlan RAG API 서비스에 오신 것을 환영합니다!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test-embedding")
async def test_embedding():
    """임베딩 모델 테스트 엔드포인트"""
    if embeddings_instance is None:
        from app.services.embeddings import get_embeddings_instance
        embeddings = get_embeddings_instance()
    else:
        embeddings = embeddings_instance
    
    # 테스트 텍스트 임베딩
    test_text = "법률 자문 시스템 테스트"
    vector = embeddings.embed_query(test_text)
    
    return {
        "text": test_text,
        "vector_size": len(vector),
        "vector_preview": vector[:5]  # 처음 5개 요소만 표시
    }

@app.get("/test-llm")
async def test_llm():
    """LLM 모델 테스트 엔드포인트"""
    if llm_instance is None:
        from app.services.llm import get_llm_instance
        llm = get_llm_instance()
    else:
        llm = llm_instance
    
    # 테스트 프롬프트
    test_prompt = "법률 자문 시스템에 대해 간단히 설명해주세요."
    response = llm.invoke(test_prompt)
    
    return {
        "prompt": test_prompt,
        "response": response.content
    }

# 입력 모델
class QueryRequest(BaseModel):
    query: str
    law_type: str = "all"  # 기본값은 모든 법률 분야


@app.get("/test-loader")
async def test_document_loader():
    """문서 로더 테스트 엔드포인트"""
    try:
        from app.utils.helpers import load_commercial_law
        
        # 문서 로드
        documents = load_commercial_law()
        
        # 결과 반환
        return {
            "status": "success",
            "total_documents": len(documents),
            "first_document_preview": {
                "content": documents[0].page_content[:200] + "..." if documents else "",
                "metadata": documents[0].metadata if documents else {}
            }
        }
    except Exception as e:
        logger.error(f"문서 로더 테스트 실패: {str(e)}")
        return {
            "status": "error", 
            "error": str(e)
        }

from fastapi.responses import StreamingResponse
import json

@app.post("/api/query-stream")
async def query_rag_stream(request: QueryRequest):
    """RAG 스트리밍 질의응답 API"""
    logger.info(f"스트리밍 쿼리 요청: {request.query}")
    
    def generate_response():  # async 제거
        try:
            # RAG 파이프라인 가져오기
            from app.services.rag import get_rag_pipeline
            rag = get_rag_pipeline()
            
            # 법률 분야 변경 필요시
            if request.law_type != rag.law_type:
                logger.info(f"법률 분야 변경: {rag.law_type} -> {request.law_type}")
                from app.services.rag import init_rag_pipeline
                rag = init_rag_pipeline(request.law_type)
            
            # 스트리밍 응답 생성
            for chunk in rag.invoke_streaming(request.query):
                # Server-Sent Events 형식으로 전송
                yield f"data: {json.dumps({'chunk': chunk, 'status': 'generating'}, ensure_ascii=False)}\n\n"
            
            # 완료 신호
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
            
        except Exception as e:
            logger.error(f"스트리밍 쿼리 처리 중 오류: {str(e)}")
            yield f"data: {json.dumps({'error': str(e), 'status': 'error'})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",  # 이 부분 수정!
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        }
    )

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def api_health_check():
    return {"status": "healthy"}