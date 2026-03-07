"""
5단계: 텔레그램 봇 연동
- python-telegram-bot으로 텔레그램 인터페이스
- LangGraph 에이전트를 텔레그램에서 사용
- chat_id 기반 세션 분리
"""
import os
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes


# === 커스텀 도구 (4단계와 동일) ===

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


# === 에이전트 구성 ===

llm = ChatOllama(model="qwen3:4b")
memory = MemorySaver()
tools = [get_current_time, calculate, search_memo]
agent = create_react_agent(llm, tools=tools, checkpointer=memory)


# === 텔레그램 핸들러 ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """"/start 명령 처리"""
    await update.message.reply_text(
        "안녕! 도구 장착 AI 봇이야.\n"
        f"사용 가능한 도구: {[t.name for t in tools]}\n"
        "아무 메시지나 보내봐."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """일반 메시지 처리 — LangGraph 에이전트에 전달"""
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)

    # chat_id를 thread_id로 사용 → 채팅방별 세션 분리
    config = {"configurable": {"thread_id": chat_id}}

    response = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config
    )

    ai_message = response["messages"][-1]
    await update.message.reply_text(ai_message.content)


# === 봇 실행 ===

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

app = Application.builder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("텔레그램 봇 시작...")
app.run_polling()
