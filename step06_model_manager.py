"""
6단계: ModelManager 구현
- 여러 LLM 프로바이더를 동적으로 전환
- /model 명령으로 텔레그램에서 모델 변경
- 환경변수로 API 키 관리
"""
import os
from datetime import datetime
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
def search_memo(keyword: str) -> str:
    """키워드로 메모를 검색한다."""
    memos = {
        "회의": "3월 10일 오후 2시 팀 회의",
        "쇼핑": "우유, 계란, 빵 사기",
        "운동": "매일 저녁 30분 러닝",
    }
    for key, value in memos.items():
        if keyword in key:
            return f"메모 발견: {value}"
    return f"'{keyword}' 관련 메모 없음"


TOOLS = [get_current_time, calculate, search_memo]


# === ModelManager ===

class ModelManager:
    def __init__(self):
        self.models = {
            "ollama": lambda: ChatOllama(model="qwen3:4b"),
            "gemini": lambda: ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.environ["GOOGLE_API_KEY"],
            ),
        }
        self.current = "ollama"
        self.memory = MemorySaver()
        self.agent = self._build_agent()

    def _build_agent(self):
        llm = self.models[self.current]()
        return create_react_agent(llm, tools=TOOLS, checkpointer=self.memory)

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
        return "사용 가능한 모델:\n" + "\n".join(lines)


manager = ModelManager()


# === 텔레그램 핸들러 ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI 봇 (멀티모델)\n"
        "명령어:\n"
        "/model list — 모델 목록\n"
        "/model <이름> — 모델 전환"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(manager.list_models())
        return

    sub = args[0]
    if sub == "list":
        await update.message.reply_text(manager.list_models())
    else:
        result = manager.switch(sub)
        await update.message.reply_text(result)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)
    config = {"configurable": {"thread_id": chat_id}}

    response = manager.agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config
    )

    ai_message = response["messages"][-1]
    await update.message.reply_text(ai_message.content)


# === 봇 실행 ===

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("model", model_command))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print(f"봇 시작 (현재 모델: {manager.current})")
app.run_polling()
