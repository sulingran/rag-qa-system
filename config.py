import os

# DashScope / OpenAI-compatible API
DASHSCOPE_API_KEY = "your-api-key-here"  # 替换为你的DashScope API Key
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v3"

# Storage paths (relative to backend/)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Text splitter
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
MAX_RETRIEVED_DOCS = 4

# Conversation history: keep last N turns
MAX_HISTORY_TURNS = 5
