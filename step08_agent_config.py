"""
8단계: 마크다운 기반 에이전트 설정
- SOUL.md, USER.md, TOOLS.md, MEMORY.md로 봇 성격/행동 관리
- 시스템 프롬프트에 자동 조합
- /soul, /user, /memory 명령으로 텔레그램에서 조회/편집
"""
import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()

CONFIG_DIR = Path(__file__).parent / "config"


# === 설정 파일 관리 ===

class AgentConfig:
    """마크다운 파일 기반 에이전트 설정"""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
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

    def write(self, name: str, content: str):
        path = self.files.get(name)
        if path:
            path.write_text(content, encoding="utf-8")

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


TOOLS = [get_current_time, calculate]


# === ModelManager (7단계 기반 + 시스템 프롬프트) ===

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
        """설정 변경 후 에이전트 재구성"""
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
        mode = "자동" if self.auto_mode else "수동"
        return f"모드: {mode}\n" + "\n".join(lines)

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
        "AI 봇 (설정 가능)\n"
        "/soul — 페르소나 조회/편집\n"
        "/user — 사용자 정보 조회/편집\n"
        "/memory — 장기 메모리 조회\n"
        "/model list — 모델 목록\n"
        "/model <이름> — 모델 전환"
    )


async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE, name: str):
    """설정 파일 조회/편집 공통 핸들러"""
    args = context.args
    if not args:
        # 조회
        content = config.read(name)
        await update.message.reply_text(content or "(비어있음)")
    else:
        # 편집 (인자 전체를 새 내용으로)
        new_content = " ".join(args)
        config.write(name, new_content)
        manager.rebuild()
        await update.message.reply_text(f"{name.upper()}.md 업데이트 완료")


async def soul_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await config_command(update, context, "soul")


async def user_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await config_command(update, context, "user")


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """메모리는 조회만 (편집은 봇이 자동으로)"""
    content = config.read("memory")
    await update.message.reply_text(content or "(비어있음)")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or args[0] == "list":
        await update.message.reply_text(manager.list_models())
    else:
        result = manager.switch(args[0])
        await update.message.reply_text(result)


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
app.add_handler(CommandHandler("soul", soul_command))
app.add_handler(CommandHandler("user", user_command))
app.add_handler(CommandHandler("memory", memory_command))
app.add_handler(CommandHandler("model", model_command))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print(f"봇 시작 (시스템 프롬프트 길이: {len(config.build_system_prompt())}자)")
app.run_polling()
