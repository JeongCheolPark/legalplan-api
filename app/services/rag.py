import logging
from typing import List, Dict, Any
from app.core.config import settings
from app.services.embeddings import get_embeddings_instance
from app.services.llm import get_llm_instance
from app.services.vectorstore import get_vector_stores

# 로거 설정
logger = logging.getLogger(__name__)

def format_docs(docs):
    """검색된 문서를 문자열로 포맷팅"""
    return "\n\n".join(doc.page_content for doc in docs)

class RagPipeline:
    """간단한 가상 RAG 파이프라인 구현"""
    
    def __init__(self, law_type="all"):
        self.law_type = law_type
        logger.info(f"RagPipeline 초기화: {law_type}")
        
        # 벡터 스토어, 임베딩, LLM 가져오기
        self.vector_stores = get_vector_stores()
        self.embeddings = get_embeddings_instance()
        self.llm = get_llm_instance()
    
    def _get_combined_retriever(self, query):
        """여러 벡터 스토어를 조합한 리트리버 구현"""
        # law_type이 "all"이면 모든 벡터 스토어 사용
        if self.law_type == "all":
            all_docs = []
            
            # 각 벡터 스토어에서 검색
            for name, vector_store in self.vector_stores.items():
                if vector_store is not None:  # None 체크 추가
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 2}  # 각 벡터 스토어에서 2개씩 가져옴
                    )
                    docs = retriever.get_relevant_documents(query)
                    logger.info(f"{name}에서 {len(docs)}개 문서 검색됨")
                    all_docs.extend(docs)
                else:
                    logger.info(f"{name}은 아직 구현되지 않음 (None)")
            
            return all_docs
        else:
            # 특정 법률 분야 벡터 스토어만 사용
            vector_store = self.vector_stores.get(f"{self.law_type}_law")
            if not vector_store:
                logger.warning(f"{self.law_type}_law 벡터 스토어가 없습니다")
                return []
            
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            return retriever.get_relevant_documents(query)
    
    def invoke_streaming(self, query):
        """RAG 파이프라인 스트리밍 실행"""
        logger.info(f"RAG 스트리밍 파이프라인 실행: {query[:30]}...")
        
        # 1. 관련 문서 검색
        docs = self._get_combined_retriever(query)
        logger.info(f"총 {len(docs)}개 문서 검색됨")
        
        if not docs:
            yield "관련 문서를 찾을 수 없습니다."
            return
        
        # 2. 문서 컨텍스트 포맷팅
        context = format_docs(docs)
        
        # 3. 프롬프트 구성
        system_template = """당신은 한국 법률 전문 AI 어시스턴트입니다. 
다음 법률 문서 컨텍스트를 참고하여 사용자의 법률 관련 질문에 정확하고 도움이 되는 답변을 제공하세요.
컨텍스트에 관련 정보가 없는 경우, "제가 가진 정보로는 답변드리기 어렵습니다"라고 정직하게 답변하세요.

컨텍스트:
{context}

사용자 질문: {question}

답변 작성 지침:
1. 법률 용어는 정확하게 사용하세요.
2. 관련 법조항이 있다면 명시하세요.
3. 복잡한 개념은 쉽게 풀어서 설명하세요.
4. 확실하지 않은 내용에 대해서는 단정적으로 답변하지 마세요.
5. 법률 조언이 아닌 정보 제공 차원의 답변임을 명시하세요.
"""
    
        prompt = system_template.format(context=context, question=query)
        
        # 4. 스트리밍 LLM 호출
        from app.services.llm import get_streaming_llm
        streaming_llm = get_streaming_llm()
        
        if streaming_llm is None:
            yield "LLM 서비스를 사용할 수 없습니다."
            return
        
        # 5. 스트리밍 응답 생성
        try:
            for chunk in streaming_llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"스트리밍 응답 생성 오류: {str(e)}")
            yield f"응답 생성 중 오류가 발생했습니다: {str(e)}"



    def invoke(self, query):
        """RAG 파이프라인 실행"""
        logger.info(f"RAG 파이프라인 실행: {query[:30]}...")
        
        # 1. 관련 문서 검색
        docs = self._get_combined_retriever(query)
        logger.info(f"총 {len(docs)}개 문서 검색됨")
        
        if not docs:
            return "관련 문서를 찾을 수 없습니다."
        
        # 2. 문서 컨텍스트 포맷팅
        context = format_docs(docs)
        
        # 3. 프롬프트 구성
        system_template = """당신은 한국 법률 전문 AI 어시스턴트입니다. 
다음 법률 문서 컨텍스트를 참고하여 사용자의 법률 관련 질문에 정확하고 도움이 되는 답변을 제공하세요.
컨텍스트에 관련 정보가 없는 경우, "제가 가진 정보로는 답변드리기 어렵습니다"라고 정직하게 답변하세요.

컨텍스트:
{context}

사용자 질문: {question}

답변 작성 지침:
1. 법률 용어는 정확하게 사용하세요.
2. 관련 법조항이 있다면 명시하세요.
3. 복잡한 개념은 쉽게 풀어서 설명하세요.
4. 확실하지 않은 내용에 대해서는 단정적으로 답변하지 마세요.
5. 법률 조언이 아닌 정보 제공 차원의 답변임을 명시하세요.
"""
        
        prompt = system_template.format(context=context, question=query)
         
        # 4. 스트리밍 LLM 호출
        from app.services.llm import get_streaming_llm
        streaming_llm = get_streaming_llm()
        
        if streaming_llm is None:
            yield "LLM 서비스를 사용할 수 없습니다."
            return
        
        # 5. 스트리밍 응답 생성
        try:
            for chunk in streaming_llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"스트리밍 응답 생성 오류: {str(e)}")
            yield f"응답 생성 중 오류가 발생했습니다: {str(e)}"

        
# RAG 파이프라인 인스턴스
rag_pipeline = None

def init_rag_pipeline(law_type="all"):
    """RAG 파이프라인 초기화"""
    global rag_pipeline
    rag_pipeline = RagPipeline(law_type)
    return rag_pipeline

def get_rag_pipeline():
    """현재 RAG 파이프라인 인스턴스 반환 또는 초기화"""
    global rag_pipeline
    if rag_pipeline is None:
        return init_rag_pipeline()
    return rag_pipeline