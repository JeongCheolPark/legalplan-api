import json
import os
from typing import List, Dict, Any
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class CommercialLawLoader:
    """상법 JSON 파일을 로드하여 LangChain Document 객체로 변환하는 클래스"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self) -> List[Document]:
        """JSON 파일을 로드하여 Document 객체 리스트로 변환"""
        logger.info(f"상법 데이터 로드 시작: {self.file_path}")
        
        # 파일 존재 확인
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")
        
        # JSON 파일 로드
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"JSON 파일 로드 성공: {len(data)}개 조문")
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패: {str(e)}")
            raise
        
        # Document 객체 리스트 생성
        documents = []
        
        for item in data:
            try:
                # 페이지 내용: 제목 + 내용만 간단하게
                page_content = f"{item.get('제목', '')} {item.get('내용', '')}"

                # 메타데이터를 최소한으로 간소화
                metadata = {
                    "source": "commercial-law.json",
                    "law_type": "commercial", 
                    "article": item.get('조문', ''),
                    "title": item.get('제목', '')
                }

                       
                # Document 객체 생성
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"조문 처리 중 오류 (건너뜀): {str(e)}")
                continue
        
        logger.info(f"Document 객체 변환 완료: {len(documents)}개")
        return documents

def load_commercial_law(file_path: str = "./data/commercial_law/commercial-law.json") -> List[Document]:
    """상법 데이터를 로드하는 편의 함수"""
    loader = CommercialLawLoader(file_path)
    return loader.load()