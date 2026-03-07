"""
10단계: 장기 메모리 — SQLite 대화 이력 + 요약
- SqliteSaver로 대화 이력 영구 저장
- 봇 재시작해도 이전 대화 유지
- 대화 길어지면 자동 요약
"""
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()

CONFIG_DIR = Path(__file__).parent / "config"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
DB_PATH = Path(__file__).parent / "chat_history.db"

MAX_MESSAGES = 20  # 이 이상이면 요약


# === 벡터 DB ===

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="knowledge",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


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
    """수학 계산식을 받아서 결과를 반환한다."""
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


# === 대화 요약 ===

def summarize_messages(messages: list, llm) -> str:
    """대화 이력을 요약한다."""
    conversation = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation.append(f"사용자: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation.append(f"AI: {msg.content}")

    summary_prompt = f"""다음 대화를 3-5문장으로 요약해줘. 중요한 정보와 맥락을 유지해:

{chr(10).join(conversation)}

요약:"""

    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return response.content


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
        self.memory = SqliteSaver.from_conn_string(str(DB_PATH))
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

    def get_llm(self):
        return self.models[self.current]()

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

    def check_and_summarize(self, thread_id: str):
        """대화가 길어지면 요약"""
        cfg = {"configurable": {"thread_id": thread_id}}
        state = self.agent.get_state(cfg)
        messages = state.values.get("messages", [])

        if len(messages) > MAX_MESSAGES:
            # 오래된 메시지 요약
            old_messages = messages[:-10]  # 최근 10개 제외
            summary = summarize_messages(old_messages, self.get_llm())

            # 요약 + 최근 메시지로 교체
            new_messages = [
                SystemMessage(content=f"이전 대화 요약:\n{summary}"),
                *messages[-10:]
            ]
            self.agent.update_state(cfg, {"messages": new_messages})


manager = ModelManager(config)


# === 텔레그램 핸들러 ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI 봇 (장기 메모리)\n"
        "/model list|<이름> — 모델 관리\n"
        "/kb add|search|count — 지식베이스\n"
        "/history — 대화 이력 정보\n"
        "/reset — 대화 초기화"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or args[0] == "list":
        await update.message.reply_text(manager.list_models())
    else:
        result = manager.switch(args[0])
        await update.message.reply_text(result)


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
        await update.message.reply_text(f"{len(chunks)}개 청크 추가")
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


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}
    state = manager.agent.get_state(cfg)
    messages = state.values.get("messages", [])
    await update.message.reply_text(f"저장된 메시지: {len(messages)}개\nDB: {DB_PATH.name}")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}
    manager.agent.update_state(cfg, {"messages": []})
    await update.message.reply_text("대화 이력 초기화 완료")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}

    response_text, used_model = manager.invoke_with_fallback(
        {"messages": [HumanMessage(content=user_message)]},
        cfg
    )

    # 대화 길어지면 요약
    manager.check_and_summarize(chat_id)

    suffix = f"\n[{used_model}]" if manager.auto_mode else ""
    await update.message.reply_text(response_text + suffix)


# === 봇 실행 ===

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("model", model_command))
app.add_handler(CommandHandler("kb", kb_command))
app.add_handler(CommandHandler("history", history_command))
app.add_handler(CommandHandler("reset", reset_command))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print(f"봇 시작 (DB: {DB_PATH})")
app.run_polling()
