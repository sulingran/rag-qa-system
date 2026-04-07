企业文档智能问答系统
基于 LangChain + Chroma + Flask 构建的 RAG（检索增强生成）全栈问答系统，支持上传企业文档并通过自然语言提问获取答案。
功能特性

支持 PDF / TXT / DOCX 多格式文档上传
基于向量语义检索，精准匹配相关段落
调用 Qwen 大模型生成自然语言答案
答案附带来源文件名、页码、段落编号，可溯源
支持多轮对话，保留上下文记忆
前后端分离，原生 HTML + JS 界面

技术栈
模块技术LLMQwen-Plus（阿里云DashScope）Embeddingtext-embedding-v3向量数据库ChromaRAG框架LangChain后端Flask前端原生 HTML / CSS / JS
项目结构
rag-qa-system/
├── backend/
│   ├── app.py                 # Flask主程序，API接口
│   ├── config.py              # 配置文件
│   ├── document_processor.py  # 文档解析与分块
│   ├── vector_store.py        # Chroma向量库操作
│   ├── qa_chain.py            # 问答链与对话记忆
│   └── requirements.txt       # 依赖列表
└── frontend/
    └── index.html             # 前端页面
快速开始
安装依赖
cd backend
pip install -r requirements.txt
配置 API Key，在 config.py 中填入你的阿里云DashScope API Key
DASHSCOPE_API_KEY = "your-api-key-here"
启动服务
python app.py
打开浏览器访问 http://localhost:5000
系统架构
用户上传文档
    ↓
document_processor.py 解析 + 分块
    ↓
vector_store.py Embedding向量化 → Chroma存储
    ↓
用户提问 → 语义检索Top-K段落
    ↓
qa_chain.py 拼接Prompt → Qwen生成答案
    ↓
返回答案 + 来源段落
