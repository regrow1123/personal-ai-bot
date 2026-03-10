"""
14단계: 역할별 에이전트 분리 + LangGraph 협업 그래프
- 라우터가 질문 유형을 분류
- 유형별 전문 에이전트로 라우팅
- LangGraph StateGraph로 흐름 제어
"""
import os
import asyncio
import operator
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from faster_whisper import WhisperModel
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()

CONFIG_DIR = Path(__file__).parent / "config"
DB_PATH = Path(__file__).parent / "chat_history.db"


# === STT ===

whisper_model = None

def get_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    return whisper_model


def transcribe_audio(audio_path: str) -> str:
    model = get_whisper()
    segments, info = model.transcribe(audio_path, language="ko")
    text = "".join(segment.text for segment in segments)
    return text.strip()


# === 설정 파일 ===

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


# === 도구 ===

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
def run_python(code: str) -> str:
    """파이썬 코드를 실행하고 결과를 반환한다."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        if "_result" in exec_globals:
            return str(exec_globals["_result"])
        return "실행 완료 (출력 없음)"
    except Exception as e:
        return f"실행 오류: {e}"


# === LLM 팩토리 ===

def get_llm(provider: str = "ollama"):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.environ["GOOGLE_API_KEY"],
        )
    return ChatOllama(model="qwen3:4b")


# === 그래프 상태 정의 ===

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    category: str
    current_agent: str
    response: str


# === 노드 함수 ===

def router_node(state: AgentState) -> dict:
    """질문 유형 분류"""
    last_message = state["messages"][-1].content

    llm = get_llm("ollama")
    classification = llm.invoke([
        SystemMessage(content=(
            "사용자 메시지를 다음 중 하나로 분류해. 카테고리 이름만 답해:\n"
            "- chat: 일반 대화, 인사, 잡담\n"
            "- tool: 시간 확인, 계산, 정보 조회\n"
            "- code: 코드 작성, 프로그래밍 질문, 디버깅\n"
            "- knowledge: 개념 설명, 학습 질문, 지식 검색"
        )),
        HumanMessage(content=last_message),
    ])

    category = classification.content.strip().lower()
    # 유효하지 않은 카테고리면 chat으로 폴백
    if category not in ["chat", "tool", "code", "knowledge"]:
        category = "chat"

    return {"category": category, "current_agent": "router"}


def chat_agent(state: AgentState) -> dict:
    """일반 대화 에이전트"""
    system = config.build_system_prompt()
    llm = get_llm("ollama")

    response = llm.invoke([
        SystemMessage(content=system + "\n\n친근하고 자연스럽게 대화해."),
        *state["messages"],
    ])
    return {
        "messages": [response],
        "response": response.content,
        "current_agent": "chat",
    }


def tool_agent(state: AgentState) -> dict:
    """도구 사용 에이전트"""
    llm = get_llm("ollama")
    llm_with_tools = llm.bind_tools([get_current_time, calculate])

    response = llm_with_tools.invoke([
        SystemMessage(content="도구를 사용해서 정확한 정보를 제공해."),
        *state["messages"],
    ])

    # 도구 호출이 있으면 실행
    if response.tool_calls:
        tool_map = {"get_current_time": get_current_time, "calculate": calculate}
        tool_results = []
        for tc in response.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
                tool_results.append(f"{tc['name']}: {result}")

        # 도구 결과를 포함해서 최종 답변 생성
        final = llm.invoke([
            SystemMessage(content="도구 실행 결과를 바탕으로 자연스럽게 답변해."),
            *state["messages"],
            AIMessage(content=f"도구 결과: {'; '.join(tool_results)}"),
        ])
        return {
            "messages": [final],
            "response": final.content,
            "current_agent": "tool",
        }

    return {
        "messages": [response],
        "response": response.content,
        "current_agent": "tool",
    }


def code_agent(state: AgentState) -> dict:
    """코딩 에이전트"""
    llm = get_llm("gemini")  # 코딩은 Gemini가 더 잘함

    response = llm.invoke([
        SystemMessage(content=(
            "너는 프로그래밍 전문가야. "
            "코드 작성, 디버깅, 설명을 해줘. "
            "코드는 반드시 코드 블록으로 감싸줘."
        )),
        *state["messages"],
    ])
    return {
        "messages": [response],
        "response": response.content,
        "current_agent": "code",
    }


def knowledge_agent(state: AgentState) -> dict:
    """지식/학습 에이전트"""
    llm = get_llm("gemini")  # 지식 질문은 Gemini가 더 잘함

    response = llm.invoke([
        SystemMessage(content=(
            "너는 지식 전문가야. "
            "개념을 명확하고 구조적으로 설명해줘. "
            "비유와 예시를 활용해."
        )),
        *state["messages"],
    ])
    return {
        "messages": [response],
        "response": response.content,
        "current_agent": "knowledge",
    }


# === 조건 분기 ===

def route_by_category(state: AgentState) -> Literal["chat", "tool", "code", "knowledge"]:
    return state["category"]


# === 그래프 구성 ===

def build_graph():
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("router", router_node)
    graph.add_node("chat", chat_agent)
    graph.add_node("tool", tool_agent)
    graph.add_node("code", code_agent)
    graph.add_node("knowledge", knowledge_agent)

    # 흐름 정의
    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_by_category,
        {
            "chat": "chat",
            "tool": "tool",
            "code": "code",
            "knowledge": "knowledge",
        }
    )

    graph.add_edge("chat", END)
    graph.add_edge("tool", END)
    graph.add_edge("code", END)
    graph.add_edge("knowledge", END)

    memory = SqliteSaver.from_conn_string(str(DB_PATH))
    return graph.compile(checkpointer=memory)


# === 텔레그램 핸들러 ===

agent_graph = build_graph()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI 봇 (멀티 에이전트)\n"
        "자동으로 질문 유형을 분류해서 전문 에이전트가 답변합니다\n\n"
        "- 일반 대화 → 채팅 에이전트\n"
        "- 시간/계산 → 도구 에이전트\n"
        "- 코드/프로그래밍 → 코딩 에이전트\n"
        "- 개념/학습 → 지식 에이전트\n\n"
        "/reset — 대화 초기화"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}
    agent_graph.update_state(cfg, {"messages": []})
    await update.message.reply_text("대화 초기화 완료")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}

    try:
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            cfg,
        )

        response_text = result.get("response", "응답을 생성하지 못했습니다.")
        agent_name = result.get("current_agent", "unknown")
        category = result.get("category", "unknown")

        await update.message.reply_text(f"{response_text}\n[{agent_name} | {category}]")

    except Exception as e:
        await update.message.reply_text(f"오류: {e}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("이미지 분석 중...")
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()
    caption = update.message.caption or "이 이미지를 분석해서 설명해줘."

    try:
        b64 = base64.b64encode(bytes(image_bytes)).decode()
        message = HumanMessage(content=[
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ])
        vision_llm = get_llm("gemini")
        response = vision_llm.invoke([message])
        await update.message.reply_text(f"{response.content}\n[vision | image]")
    except Exception as e:
        await update.message.reply_text(f"이미지 분석 실패: {e}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("음성 인식 중...")
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

    try:
        text = transcribe_audio(tmp_path)
        if not text:
            await update.message.reply_text("음성을 인식하지 못했습니다.")
            return

        chat_id = str(update.effective_chat.id)
        cfg = {"configurable": {"thread_id": chat_id}}

        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=text)]}, cfg
        )
        response_text = result.get("response", "응답 실패")
        agent_name = result.get("current_agent", "unknown")

        await update.message.reply_text(
            f"[음성 인식] {text}\n\n{response_text}\n[{agent_name}]"
        )
    except Exception as e:
        await update.message.reply_text(f"음성 처리 실패: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def main():
    BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("봇 시작 (멀티 에이전트)")
    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
