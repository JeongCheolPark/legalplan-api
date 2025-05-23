import logging
from typing import List, Dict, Any
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import time

from app.core.config import settings
from app.services.embeddings import get_embeddings_instance
from app.utils.helpers import load_commercial_law

# 로거 설정
logger = logging.getLogger(__name__)

# 전역 Qdrant 클라이언트
qdrant_client = None
vector_stores = {}

def init_qdrant_client():
    """Qdrant 클라이언트 초기화"""
    global qdrant_client
    
    if qdrant_client is None:
        logger.info("Qdrant 클라이언트 초기화 중...")
        # 로컬 디스크 기반 Qdrant 사용
        qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
        logger.info(f"Qdrant 클라이언트 초기화 완료: {settings.QDRANT_PATH}")
    
    return qdrant_client

def create_commercial_law_vectorstore():
    """상법 벡터 스토어 생성 및 데이터 로드 (수동 방식)"""
    logger.info("상법 벡터 스토어 생성 시작...")
    
    # Qdrant 클라이언트 초기화
    client = init_qdrant_client()
    
    # 임베딩 모델 가져오기
    embeddings = get_embeddings_instance()
    
    # 상법 문서 로드
    logger.info("상법 문서 로드 중...")
    documents = load_commercial_law()
    logger.info(f"{len(documents)}개 상법 문서 로드 완료")
    
    # 소량 테스트 (처음 10개 문서만)
    test_documents = documents[:10]
    logger.info(f"테스트용으로 {len(test_documents)}개 문서 사용")
    
    # 컬렉션명
    collection_name = "commercial_law"
    
    try:
        # 기존 컬렉션 삭제 (있다면)
        try:
            client.delete_collection(collection_name)
            logger.info(f"기존 '{collection_name}' 컬렉션 삭제됨")
        except:
            pass  # 컬렉션이 없으면 무시
        
        # 임베딩 벡터 생성
        logger.info("문서 임베딩 생성 중...")
        start_time = time.time()
        
        texts = [doc.page_content for doc in test_documents]
        vectors = embeddings.embed_documents(texts)
        
        end_time = time.time()
        logger.info(f"임베딩 생성 완료 (소요 시간: {end_time - start_time:.2f}초)")
        
        # 컬렉션 생성
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
        )
        logger.info(f"컬렉션 '{collection_name}' 생성 완료")
        
        # 포인트 생성 및 업로드
        from qdrant_client.models import PointStruct
        
        points = []
        for i, (doc, vector) in enumerate(zip(test_documents, vectors)):
            point = PointStruct(
                id=i,
                vector=vector,
                payload={
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "law_type": doc.metadata.get("law_type", ""),
                    "article": doc.metadata.get("article", ""),
                    "title": doc.metadata.get("title", "")
                }
            )
            points.append(point)
        
        # Qdrant에 포인트 업로드
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"{len(points)}개 포인트 업로드 완료")
        
        # QdrantVectorStore 객체 생성
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        
        logger.info("벡터 스토어 생성 완료")
        return vectorstore
        
    except Exception as e:
        logger.error(f"벡터 스토어 생성 실패: {str(e)}")
        raise

def init_vector_stores():
    """벡터 스토어 초기화 (상법만)"""
    global vector_stores
    
    try:
        # 상법 벡터 스토어만 실제 구현
        logger.info("실제 상법 벡터 스토어 초기화 중...")
        commercial_vectorstore = create_commercial_law_vectorstore()
        vector_stores["commercial_law"] = commercial_vectorstore
        logger.info("상법 벡터 스토어 초기화 완료")
        
        # 나머지는 임시로 None (나중에 구현)
        vector_stores["civil_law"] = None
        vector_stores["criminal_law"] = None
        
        return vector_stores
        
    except Exception as e:
        logger.error(f"벡터 스토어 초기화 실패: {str(e)}")
        # 실패 시 빈 딕셔너리 반환
        return {}

def get_vector_stores():
    """현재 벡터 스토어 인스턴스들 반환 또는 초기화"""
    global vector_stores
    if not vector_stores:
        return init_vector_stores()
    return vector_stores