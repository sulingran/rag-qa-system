import os
import uuid

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from config import UPLOAD_FOLDER
from document_processor import process_document, SUPPORTED_EXTENSIONS
from vector_store import add_documents, delete_documents_by_source
from qa_chain import ConversationManager, answer_question

# ─────────────────── app setup ───────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory state (single-user demo)
conversation = ConversationManager()
uploaded_docs: dict[str, dict] = {}   # { doc_id: {filename, chunks, path} }


# ─────────────────── static ───────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ─────────────────── documents ───────────────────

@app.route("/api/documents", methods=["GET"])
def list_documents():
    result = [
        {"doc_id": did, "filename": info["filename"], "chunks": info["chunks"]}
        for did, info in uploaded_docs.items()
    ]
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "请选择要上传的文件"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400

    filename: str = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"不支持 {ext} 格式，请上传 PDF / TXT / DOCX"}), 400

    # Check duplicate filename
    for info in uploaded_docs.values():
        if info["filename"] == filename:
            return jsonify({"error": f"文件 [{filename}] 已上传，请先删除旧文件再重新上传"}), 400

    doc_id = uuid.uuid4().hex[:8]
    save_path = os.path.join(UPLOAD_FOLDER, f"{doc_id}_{filename}")
    file.save(save_path)

    try:
        chunks = process_document(save_path, filename)
        add_documents(chunks)
        uploaded_docs[doc_id] = {
            "filename": filename,
            "chunks": len(chunks),
            "path": save_path,
        }
        return jsonify({"doc_id": doc_id, "filename": filename, "chunks": len(chunks)})
    except Exception as exc:
        # Clean up saved file on failure
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id: str):
    if doc_id not in uploaded_docs:
        return jsonify({"error": "文档不存在"}), 404

    info = uploaded_docs.pop(doc_id)
    delete_documents_by_source(info["filename"])
    if os.path.exists(info["path"]):
        os.remove(info["path"])

    return jsonify({"success": True, "filename": info["filename"]})


# ─────────────────── chat ───────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    if not uploaded_docs:
        return jsonify({"error": "请先上传至少一份文档再提问"}), 400

    try:
        answer, sources = answer_question(question, conversation)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as exc:
        return jsonify({"error": f"问答失败：{exc}"}), 500


@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    conversation.clear()
    return jsonify({"success": True})


# ─────────────────── run ───────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
