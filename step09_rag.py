"""
9단계: RAG — ChromaDB 기반 개인 지식베이스
- 문서를 벡터 DB에 저장
- 시맨틱 검색으로 관련 문서 검색
- 검색 결과를 LLM 프롬프트에 주입
"""
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()

CONFIG_DIR = Path(__file__).parent / "config"
CHROMA_DIR = Path(__file__).parent / "chroma_db"


# === 벡터 DB 설정 ===

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="knowledge",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


# === 설정 파일 관리 ===

class AgentConfig:
    def __init__(self, config_dir: Path):
        self.files = {
            "soul": config_dir / "SOUL.md",
            "user": config_dir / "USER.md",
            "tools": config_dir / "TOOLS.md",
            "memory": config_dir / "MEMORY.md",
        }

    def read(self, name: str) -> str:
        path = self.files.get(name)
        if path and path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def append(self, name: str, content: str):
        path = self.files.get(name)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n{content}")

    def build_system_prompt(self) -> str:
        parts = []
        for name in ["soul", "user", "tools", "memory"]:
            content = self.read(name)
            if content.strip():
                parts.append(content)
        return "\n\n---\n\n".join(parts)


config = AgentConfig(CONFIG_DIR)


# === 커스텀 도구 ===

@tool
def get_current_time() -> str:
    """현재 날짜와 시간을 반환한다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """수학 계산식을 받아서 결과를 반환한다. 예: '2 + 3 * 4'"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def search_knowledge(query: str) -> str:
    """개인 지식베이스에서 관련 문서를 검색한다."""
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "관련 문서를 찾지 못했습니다."
    texts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "알 수 없음")
        texts.append(f"[{i}] ({source})\n{doc.page_content}")
    return "\n\n".join(texts)


@tool
def add_knowledge(content: str) -> str:
    """새로운 지식을 지식베이스에 추가한다."""
    chunks = text_splitter.split_text(content)
    vectorstore.add_texts(
        texts=chunks,
        metadatas=[{"source": "user_input", "added": datetime.now().isoformat()}] * len(chunks),
    )
    return f"지식베이스에 {len(chunks)}개 청크 추가 완료"


@tool
def save_memory(content: str) -> str:
    """중요한 정보를 장기 메모리에 저장한다."""
    config.append("memory", f"- {content}")
    return "메모리에 저장 완료"


TOOLS = [get_current_time, calculate, search_knowledge, add_knowledge, save_memory]


# === ModelManager ===

class ModelManager:
    def __init__(self, agent_config: AgentConfig):
        self.config = agent_config
        self.models = {
            "ollama": lambda: ChatOllama(model="qwen3:4b"),
            "gemini": lambda: ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.environ["GOOGLE_API_KEY"],
            ),
        }
        self.priority = ["ollama", "gemini"]
        self.current = "ollama"
        self.auto_mode = True
        self.memory = MemorySaver()
        self.agent = self._build_agent()

    def _build_agent(self):
        llm = self.models[self.current]()
        system_prompt = self.config.build_system_prompt()
        return create_react_agent(
            llm,
            tools=TOOLS,
            checkpointer=self.memory,
            prompt=system_prompt,
        )

    def rebuild(self):
        self.agent = self._build_agent()

    def switch(self, name: str) -> str:
        if name not in self.models:
            return f"없는 모델: {name}\n사용 가능: {', '.join(self.models.keys())}"
        self.current = name
        self.agent = self._build_agent()
        return f"모델 전환: {name}"

    def list_models(self) -> str:
        lines = []
        for name in self.models:
            marker = " (현재)" if name == self.current else ""
            lines.append(f"- {name}{marker}")
        return "\n".join(lines)

    def invoke_with_fallback(self, messages, cfg) -> tuple[str, str]:
        if not self.auto_mode:
            return self._try_invoke(self.current, messages, cfg)

        order = [self.current] + [m for m in self.priority if m != self.current]
        errors = {}

        for model_name in order:
            try:
                return self._try_invoke(model_name, messages, cfg)
            except Exception as e:
                errors[model_name] = str(e)
                continue

        error_detail = "\n".join(f"- {k}: {v}" for k, v in errors.items())
        return f"모든 모델 실패:\n{error_detail}", "none"

    def _try_invoke(self, model_name, messages, cfg) -> tuple[str, str]:
        if model_name != self.current:
            self.switch(model_name)
        response = self.agent.invoke(messages, cfg)
        ai_message = response["messages"][-1]
        return ai_message.content, model_name


manager = ModelManager(config)


# === 텔레그램 핸들러 ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI 봇 (RAG 지식베이스)\n"
        "/kb add <내용> — 지식 추가\n"
        "/kb search <쿼리> — 지식 검색\n"
        "/kb count — 저장된 문서 수\n"
        "/model list|<이름> — 모델 관리\n"
        "/memory — 장기 메모리 조회"
    )


async def kb_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("사용법: /kb add|search|count")
        return

    sub = args[0]
    rest = " ".join(args[1:])

    if sub == "add" and rest:
        chunks = text_splitter.split_text(rest)
        vectorstore.add_texts(
            texts=chunks,
            metadatas=[{"source": "telegram", "added": datetime.now().isoformat()}] * len(chunks),
        )
        await update.message.reply_text(f"지식베이스에 {len(chunks)}개 청크 추가")

    elif sub == "search" and rest:
        results = vectorstore.similarity_search(rest, k=3)
        if results:
            text = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(results, 1))
            await update.message.reply_text(text)
        else:
            await update.message.reply_text("결과 없음")

    elif sub == "count":
        count = vectorstore._collection.count()
        await update.message.reply_text(f"저장된 문서: {count}개")

    else:
        await update.message.reply_text("사용법: /kb add|search|count")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or args[0] == "list":
        await update.message.reply_text(manager.list_models())
    else:
        result = manager.switch(args[0])
        await update.message.reply_text(result)


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    content = config.read("memory")
    await update.message.reply_text(content or "(비어있음)")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}

    response_text, used_model = manager.invoke_with_fallback(
        {"messages": [HumanMessage(content=user_message)]},
        cfg
    )

    suffix = f"\n[{used_model}]" if manager.auto_mode else ""
    await update.message.reply_text(response_text + suffix)


# === 봇 실행 ===

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("kb", kb_command))
app.add_handler(CommandHandler("model", model_command))
app.add_handler(CommandHandler("memory", memory_command))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("봇 시작 (RAG 활성)")
app.run_polling()
