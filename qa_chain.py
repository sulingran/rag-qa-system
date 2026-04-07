from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from vector_store import similarity_search
from config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL,
    LLM_MODEL, MAX_RETRIEVED_DOCS, MAX_HISTORY_TURNS,
)


# ─────────────────── conversation memory ───────────────────

@dataclass
class Turn:
    human: str
    ai: str


@dataclass
class ConversationManager:
    history: list[Turn] = field(default_factory=list)

    def add_turn(self, human: str, ai: str) -> None:
        self.history.append(Turn(human=human, ai=ai))

    def recent_turns(self) -> list[Turn]:
        return self.history[-MAX_HISTORY_TURNS:]

    def history_text(self) -> str:
        lines = []
        for turn in self.recent_turns():
            lines.append(f"用户: {turn.human}")
            lines.append(f"助手: {turn.ai}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history.clear()


# ─────────────────── QA function ───────────────────

SYSTEM_PROMPT = """\
你是一个企业文档智能问答助手。请严格根据提供的【文档内容】回答用户的问题。
- 如果答案能在文档中找到，请给出准确、简洁的回答，并在回答中自然地引用来源。
- 如果文档中没有足够的相关信息，请如实告知用户，不要编造内容。
- 回答使用中文，语气专业友好。\
"""


def answer_question(
    question: str,
    conversation: ConversationManager,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks, call the LLM, update conversation memory.
    Returns (answer_text, sources_list).
    """
    # 1. Semantic retrieval
    docs = similarity_search(question, k=MAX_RETRIEVED_DOCS)

    # 2. Build context string from retrieved chunks
    context_parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        loc_parts = [f"文件：{meta.get('source', '未知')}"]
        if meta.get("page"):
            loc_parts.append(f"第 {meta['page']} 页")
        if meta.get("chunk_index"):
            loc_parts.append(f"段落 {meta['chunk_index']}")
        location = "，".join(loc_parts)
        context_parts.append(f"[来源{idx} | {location}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "（未找到相关文档片段）"

    # 3. Build user prompt
    history = conversation.history_text()
    history_section = f"\n\n【历史对话】\n{history}" if history else ""

    user_prompt = (
        f"【文档内容】\n{context}"
        f"{history_section}"
        f"\n\n【用户问题】\n{question}"
        f"\n\n请根据以上文档内容回答用户问题。"
    )

    # 4. Call LLM
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        model=LLM_MODEL,
        temperature=0.5,
    )
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    answer: str = response.content

    # 5. Update memory
    conversation.add_turn(human=question, ai=answer)

    # 6. Format sources for the frontend
    sources: list[dict] = []
    for doc in docs:
        meta = doc.metadata
        content_preview = doc.page_content
        if len(content_preview) > 250:
            content_preview = content_preview[:250] + "…"
        sources.append({
            "source": meta.get("source", "未知文件"),
            "page": meta.get("page", ""),
            "chunk_index": meta.get("chunk_index", ""),
            "preview": content_preview,
        })

    return answer, sources
