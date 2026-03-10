"""
12단계: 멀티모달 입력
- 이미지: Gemini 비전 모델로 분석
- 음성: faster-whisper STT → 텍스트 변환 → LLM 처리
"""
import os
import asyncio
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from faster_whisper import WhisperModel
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()

CONFIG_DIR = Path(__file__).parent / "config"
DB_PATH = Path(__file__).parent / "chat_history.db"


# === STT 모델 (로컬) ===

whisper_model = None

def get_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    return whisper_model


def transcribe_audio(audio_path: str) -> str:
    """음성 파일을 텍스트로 변환"""
    model = get_whisper()
    segments, info = model.transcribe(audio_path, language="ko")
    text = "".join(segment.text for segment in segments)
    return text.strip()


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


TOOLS = [get_current_time, calculate]


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

    def analyze_image(self, image_bytes: bytes, caption: str = "") -> str:
        """이미지를 비전 모델로 분석"""
        b64 = base64.b64encode(image_bytes).decode()

        prompt_text = caption if caption else "이 이미지를 분석해서 설명해줘."

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ]
        )

        # 비전은 Gemini 사용 (Ollama qwen3:4b는 비전 미지원)
        vision_llm = self.models["gemini"]()
        response = vision_llm.invoke([message])
        return response.content


manager = ModelManager(config)


# === 텔레그램 핸들러 ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI 봇 (멀티모달)\n"
        "/model list|<이름> — 모델 관리\n"
        "/reset — 대화 초기화\n\n"
        "텍스트, 이미지, 음성 메시지 모두 처리 가능"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or args[0] == "list":
        await update.message.reply_text(manager.list_models())
    else:
        result = manager.switch(args[0])
        await update.message.reply_text(result)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}
    manager.agent.update_state(cfg, {"messages": []})
    await update.message.reply_text("대화 초기화 완료")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """텍스트 메시지 처리"""
    user_message = update.message.text
    chat_id = str(update.effective_chat.id)
    cfg = {"configurable": {"thread_id": chat_id}}

    response_text, used_model = manager.invoke_with_fallback(
        {"messages": [HumanMessage(content=user_message)]},
        cfg
    )

    suffix = f"\n[{used_model}]" if manager.auto_mode else ""
    await update.message.reply_text(response_text + suffix)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """이미지 메시지 처리"""
    await update.message.reply_text("이미지 분석 중...")

    # 가장 큰 해상도 사진 다운로드
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()

    # 캡션이 있으면 질문으로 사용
    caption = update.message.caption or ""

    try:
        result = manager.analyze_image(bytes(image_bytes), caption)
        await update.message.reply_text(f"{result}\n[gemini-vision]")
    except Exception as e:
        await update.message.reply_text(f"이미지 분석 실패: {e}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """음성 메시지 처리"""
    await update.message.reply_text("음성 인식 중...")

    # 음성 파일 다운로드
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
        await file.download_to_drive(tmp_path)

    try:
        # STT 변환
        text = transcribe_audio(tmp_path)

        if not text:
            await update.message.reply_text("음성을 인식하지 못했습니다.")
            return

        await update.message.reply_text(f"인식된 텍스트: {text}")

        # 인식된 텍스트로 LLM 처리
        chat_id = str(update.effective_chat.id)
        cfg = {"configurable": {"thread_id": chat_id}}

        response_text, used_model = manager.invoke_with_fallback(
            {"messages": [HumanMessage(content=text)]},
            cfg
        )

        suffix = f"\n[{used_model}]" if manager.auto_mode else ""
        await update.message.reply_text(response_text + suffix)

    except Exception as e:
        await update.message.reply_text(f"음성 처리 실패: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def main():
    BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("봇 시작 (멀티모달: 텍스트 + 이미지 + 음성)")
    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
