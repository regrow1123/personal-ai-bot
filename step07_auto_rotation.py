"""
7단계: 동적 전환 + 자동 로테이션
- API 에러 시 자동 fallback
- 응답 시간 측정
- /auto 명령으로 자동/수동 전환
"""
import os
import time
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


TOOLS = [get_current_time, calculate]


# === ModelManager with Auto Rotation ===

class ModelManager:
    def __init__(self):
        self.models = {
            "ollama": lambda: ChatOllama(model="qwen3:4b"),
            "gemini": lambda: ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.environ["GOOGLE_API_KEY"],
            ),
        }
        self.priority = ["ollama", "gemini"]  # fallback 순서
        self.current = "ollama"
        self.auto_mode = True
        self.memory = MemorySaver()
        self.agent = self._build_agent()
        self.stats = {name: {"success": 0, "fail": 0, "total_time": 0.0} for name in self.models}

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
            s = self.stats[name]
            total = s["success"] + s["fail"]
            avg = f"{s['total_time']/s['success']:.1f}s" if s["success"] > 0 else "-"
            lines.append(f"- {name}{marker} | 성공:{s['success']} 실패:{s['fail']} 평균:{avg}")
        mode = "자동" if self.auto_mode else "수동"
        return f"모드: {mode}\n" + "\n".join(lines)

    def invoke_with_fallback(self, messages, config) -> tuple[str, str]:
        """응답 텍스트와 사용된 모델명을 반환"""
        if not self.auto_mode:
            return self._try_invoke(self.current, messages, config)

        # 현재 모델 먼저, 실패하면 나머지 순서대로
        order = [self.current] + [m for m in self.priority if m != self.current]

        for model_name in order:
            try:
                return self._try_invoke(model_name, messages, config)
            except Exception as e:
                self.stats[model_name]["fail"] += 1
                print(f"[{model_name}] 실패: {e}")
                continue

        return "모든 모델이 응답에 실패했습니다.", "none"

    def _try_invoke(self, model_name: str, messages, config) -> tuple[str, str]:
        if model_name != self.current:
            self.switch(model_name)

        start = time.time()
        response = self.agent.invoke(messages, config)
        elapsed = time.time() - start

        self.stats[model_name]["success"] += 1
        self.stats[model_name]["total_time"] += elapsed

        ai_message = response["messages"][-1]
        return ai_message.content, model_name


manager = ModelManager()


# === 텔레그램 핸들러 ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI 봇 (자동 로테이션)\n"
        "/model list — 모델 목록 + 통계\n"
        "/model <이름> — 모델 전환\n"
        "/auto — 자동 모드 토글"
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


async def auto_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    manager.auto_mode = not manager.auto_mode
    mode = "자동" if manager.auto_mode else "수동"
    await update.message.reply_text(f"모드 전환: {mode}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)
    config = {"configurable": {"thread_id": chat_id}}

    response_text, used_model = manager.invoke_with_fallback(
        {"messages": [HumanMessage(content=user_message)]},
        config
    )

    suffix = f"\n[{used_model}]" if manager.auto_mode else ""
    await update.message.reply_text(response_text + suffix)


# === 봇 실행 ===

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("model", model_command))
app.add_handler(CommandHandler("auto", auto_command))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print(f"봇 시작 (모드: 자동, 현재 모델: {manager.current})")
app.run_polling()
