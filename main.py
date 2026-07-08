"""
FastAPI 应用入口

启动命令：
    python main.py
    或
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

前端页面：
    导入侧  → http://localhost:8000/
    查询侧  → http://localhost:8000/chat
"""
from pathlib import Path

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from knowledge.processor.import_process.base import setup_logging

# 初始化日志
setup_logging()

# 创建 FastAPI 应用
app = FastAPI(title="知识库系统")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由 —— 导入侧
from knowledge.import_file_router import router as import_router
app.include_router(import_router)

# 注册路由 —— 查询侧
from knowledge.api.query_router import router as query_router
app.include_router(query_router)

# 提供前端静态文件
front_dir = Path(__file__).resolve().parent / "knowledge" / "front"
app.mount("/static", StaticFiles(directory=str(front_dir)), name="static")


@app.get("/")
async def root():
    """导入页面"""
    return FileResponse(str(front_dir / "import.html"))


@app.get("/chat")
async def chat_page():
    """聊天查询页面"""
    return FileResponse(str(front_dir / "chat.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
