import os
from typing import Optional, List
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL, CHROMA_PERSIST_DIR

# Singleton vector store
_vs: Optional[Chroma] = None


class DashScopeEmbeddings(Embeddings):
    """直接调用 OpenAI 兼容接口，确保 input 以字符串列表格式传入。"""

    def __init__(self):
        self._client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 确保每个元素是纯字符串，阿里云不接受其他类型
        clean = [str(t) for t in texts]
        response = self._client.embeddings.create(model=EMBEDDING_MODEL, input=clean)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([str(text)])[0]


def _get_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings()


def get_vector_store() -> Chroma:
    global _vs
    if _vs is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _vs = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=_get_embeddings(),
            collection_name="enterprise_docs",
        )
    return _vs


def add_documents(docs: list[Document]) -> None:
    get_vector_store().add_documents(docs)


def delete_documents_by_source(source_name: str) -> None:
    vs = get_vector_store()
    # Access the underlying chromadb collection for a direct delete-by-metadata
    vs._collection.delete(where={"source": source_name})


def similarity_search(query: str, k: int = 4) -> list[Document]:
    return get_vector_store().similarity_search(query, k=k)


def document_count() -> int:
    return get_vector_store()._collection.count()
