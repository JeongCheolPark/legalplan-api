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
                    
                    # 🔍 디버깅: 검색된 각 문서의 상세 정보 출력
                    for i, doc in enumerate(docs):
                        logger.info(f"[{name}] 문서 {i+1}:")
                        logger.info(f"  - 조문: {doc.metadata.get('article', 'N/A')}")
                        logger.info(f"  - 제목: {doc.metadata.get('title', 'N/A')}")
                        logger.info(f"  - 내용 미리보기: {doc.page_content[:100]}...")
                    
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
            docs = retriever.get_relevant_documents(query)
            
            # 🔍 디버깅: 특정 법률 검색 결과도 출력
            logger.info(f"{self.law_type}_law에서 {len(docs)}개 문서 검색됨")
            for i, doc in enumerate(docs):
                logger.info(f"문서 {i+1}:")
                logger.info(f"  - 조문: {doc.metadata.get('article', 'N/A')}")
                logger.info(f"  - 제목: {doc.metadata.get('title', 'N/A')}")
                logger.info(f"  - 내용 미리보기: {doc.page_content[:100]}...")
            
            return docs
        
    def invoke_streaming(self, query):
        """RAG 파이프라인 스트리밍 실행"""
        logger.info(f"RAG 스트리밍 파이프라인 실행: {query[:30]}...")
        
        # 🔍 디버깅: 검색 쿼리 전체 출력
        logger.info(f"[디버깅] 전체 검색 쿼리: {query}")
        
        # 1. 관련 문서 검색
        docs = self._get_combined_retriever(query)
        logger.info(f"총 {len(docs)}개 문서 검색됨")
        
        if not docs:
            yield "관련 문서를 찾을 수 없습니다."
            return
        
        # 2. 문서 컨텍스트 포맷팅
        context = format_docs(docs)
        
        # 🔍 디버깅: LLM에 전달될 컨텍스트 출력
        logger.info("[디버깅] LLM에 전달될 컨텍스트:")
        logger.info(f"컨텍스트 길이: {len(context)} 문자")
        logger.info(f"컨텍스트 내용:\n{context[:500]}...\n")  # 처음 500자만 출력
        
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
        
        # 🔍 디버깅: 최종 프롬프트 길이 출력
        logger.info(f"[디버깅] 최종 프롬프트 길이: {len(prompt)} 문자")
        
        # 4. 스트리밍 LLM 호출
        from app.services.llm import get_streaming_llm
        streaming_llm = get_streaming_llm()
        
        if streaming_llm is None:
            yield "LLM 서비스를 사용할 수 없습니다."
            return
        
        # 5. 버퍼를 사용한 스트리밍 응답 생성 (여기가 핵심 변경사항!)
        try:
            buffer = ""  # 버퍼 초기화
            
            for chunk in streaming_llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    buffer += chunk.content
                    
                    # think 태그 처리 및 전송
                    processed_content, remaining_buffer = self._process_buffer_with_think_tags(buffer)
                    buffer = remaining_buffer
                    
                    if processed_content:
                        yield processed_content
            
            # 마지막 버퍼 내용 처리
            if buffer:
                final_content = self._filter_think_tags(buffer)
                if final_content:
                    yield final_content
                    
        except Exception as e:
            logger.error(f"스트리밍 응답 생성 오류: {str(e)}")
            yield f"응답 생성 중 오류가 발생했습니다: {str(e)}"

    def _process_buffer_with_think_tags(self, buffer):
        """버퍼에서 완전한 think 태그를 찾아 처리"""
        import re
        
        # think 태그의 시작과 끝 찾기
        think_start = buffer.find('<think>')
        think_end = buffer.find('</think>')
        
        # Case 1: think 태그가 없음
        if think_start == -1:
            # 전체 버퍼를 전송하고 빈 버퍼 반환
            return buffer, ""
        
        # Case 2: think 태그가 시작했지만 끝나지 않음
        if think_start != -1 and think_end == -1:
            # think 태그 이전까지만 전송
            return buffer[:think_start], buffer[think_start:]
        
        # Case 3: 완전한 think 태그가 있음
        if think_start != -1 and think_end != -1 and think_end > think_start:
            # think 태그 이전 부분
            before_think = buffer[:think_start]
            # think 태그 이후 부분
            after_think = buffer[think_end + 8:]  # 8 = len('</think>')
            
            # 재귀적으로 처리 (여러 think 태그가 있을 수 있음)
            processed_after, remaining = self._process_buffer_with_think_tags(after_think)
            
            return before_think + processed_after, remaining
        
        # 그 외의 경우 (끝 태그만 있는 경우 등)
        return buffer, ""
 
    def _filter_think_tags(self, content):
        """<think> 태그 제거"""
        import re
        # <think>...</think> 태그와 내용 전체 제거
        filtered = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return filtered.strip()
            
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