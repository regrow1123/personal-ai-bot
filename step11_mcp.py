"""
11단계: MCP 서버 연동
- MCP 프로토콜로 외부 도구 서버와 통신
- 봇 코드 수정 없이 도구 추가/제거 가능
- 기존 @tool 도구와 MCP 도구 혼용
"""
import os
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()

CONFIG_DIR = Path(__file__).parent / "config"
DB_PATH = Path(__file__).parent / "chat_history.db"


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

    def build_system_prompt(self) -> str:
        parts = []
        for name in ["soul", "user", "tools", "memory"]:
            content = self.read(name)
            if content.strip():
                parts.append(content)
        return "\n\n---\n\n".join(parts)


config = AgentConfig(CONFIG_DIR)


# === 로컬 도구 (MCP에 없는 것) ===

@tool
def get_bot_status() -> str:
    """봇의 현재 상태 정보를 반환한다."""
    return f"봇 가동 중 | DB: {DB_PATH.name} | 시간: {datetime.now().strftime('%H:%M:%S')}"


LOCAL_TOOLS = [get_bot_status]


# === ModelManager ===

class ModelManager:
    def __init__(self, agent_config: AgentConfig, mcp_tools: list):
        self.config = agent_config
        self.all_tools = LOCAL_TOOLS + mcp_tools
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
            tools=self.all_tools,
            checkpointer=self.memory,
            prompt=system_prompt,
        )

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

    def list_tools(self) -> str:
        lines = [f"- {t.name}: {t.description}" for t in self.all_tools]
        return f"도구 ({len(self.all_tools)}개):\n" + "\n".join(lines)

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


# === 텔레그램 핸들러 ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI 봇 (MCP 연동)\n"
        "/model list|<이름> — 모델 관리\n"
        "/tools — 사용 가능한 도구 목록\n"
        "/reset — 대화 초기화"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or args[0] == "list":
        await update.message.reply_text(manager.list_models())
    else:
        result = manager.switch(args[0])
        await update.message.reply_text(result)


async def tools_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(manager.list_tools())


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}
    manager.agent.update_state(cfg, {"messages": []})
    await update.message.reply_text("대화 초기화 완료")


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


# === 메인 ===

async def main():
    global manager

    # MCP 서버 연결
    async with MultiServerMCPClient(
        {
            "personal-tools": {
                "command": "python",
                "args": [str(Path(__file__).parent / "mcp_tools_server.py")],
                "transport": "stdio",
            }
        }
    ) as mcp_client:
        mcp_tools = mcp_client.get_tools()
        print(f"MCP 도구 로드: {[t.name for t in mcp_tools]}")

        manager = ModelManager(config, mcp_tools)

        BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
        app = Application.builder().token(BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("model", model_command))
        app.add_handler(CommandHandler("tools", tools_command))
        app.add_handler(CommandHandler("reset", reset_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        print(f"봇 시작 (도구: {len(LOCAL_TOOLS)} 로컬 + {len(mcp_tools)} MCP)")
        await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
