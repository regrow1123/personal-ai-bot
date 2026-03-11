"""
15단계: 스케줄러 + 알림 (APScheduler)
- 정해진 시간에 자동 메시지 전송
- /schedule 명령으로 알림 관리
- cron, interval, 일회성 작업 지원
"""
import os
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()

CONFIG_DIR = Path(__file__).parent / "config"
DB_PATH = Path(__file__).parent / "chat_history.db"
SCHEDULES_FILE = Path(__file__).parent / "schedules.json"


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
    """수학 계산식을 받아서 결과를 반환한다."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


TOOLS = [get_current_time, calculate]


# === 스케줄 저장/로드 ===

def load_schedules() -> list[dict]:
    if SCHEDULES_FILE.exists():
        return json.loads(SCHEDULES_FILE.read_text())
    return []


def save_schedules(schedules: list[dict]):
    SCHEDULES_FILE.write_text(json.dumps(schedules, ensure_ascii=False, indent=2))


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

    def invoke_with_fallback(self, messages, cfg) -> tuple[str, str]:
        order = [self.current] + [m for m in self.priority if m != self.current]
        errors = {}
        for model_name in order:
            try:
                if model_name != self.current:
                    self.switch(model_name)
                response = self.agent.invoke(messages, cfg)
                return response["messages"][-1].content, model_name
            except Exception as e:
                errors[model_name] = str(e)
        error_detail = "\n".join(f"- {k}: {v}" for k, v in errors.items())
        return f"모든 모델 실패:\n{error_detail}", "none"


manager = ModelManager(config)


# === 스케줄러 ===

scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
bot_app = None  # 텔레그램 앱 참조


async def send_scheduled_message(chat_id: str, message: str, use_ai: bool = False):
    """스케줄된 메시지 전송"""
    if bot_app is None:
        return

    if use_ai:
        cfg = {"configurable": {"thread_id": f"schedule_{chat_id}"}}
        response, _ = manager.invoke_with_fallback(
            {"messages": [HumanMessage(content=message)]}, cfg
        )
        text = f"[예약 알림]\n{response}"
    else:
        text = f"[예약 알림] {message}"

    await bot_app.bot.send_message(chat_id=int(chat_id), text=text)


def restore_schedules():
    """저장된 스케줄 복원 (재시작 시)"""
    schedules = load_schedules()
    for s in schedules:
        job_id = s["id"]
        chat_id = s["chat_id"]
        message = s["message"]
        use_ai = s.get("use_ai", False)

        if s["type"] == "cron":
            trigger = CronTrigger(
                hour=s["hour"], minute=s["minute"], timezone="Asia/Seoul"
            )
        elif s["type"] == "interval":
            trigger = IntervalTrigger(minutes=s["minutes"])
        elif s["type"] == "once":
            run_time = datetime.fromisoformat(s["run_at"])
            if run_time < datetime.now():
                continue
            trigger = DateTrigger(run_date=run_time)
        else:
            continue

        scheduler.add_job(
            send_scheduled_message,
            trigger,
            args=[chat_id, message, use_ai],
            id=job_id,
            replace_existing=True,
        )


# === 텔레그램 핸들러 ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "AI 봇 (스케줄러)\n\n"
        "/schedule add HH:MM <메시지> — 매일 알림\n"
        "/schedule interval <분> <메시지> — 반복 알림\n"
        "/schedule once HH:MM <메시지> — 일회성 알림\n"
        "/schedule ai HH:MM <프롬프트> — 매일 AI 응답\n"
        "/schedule list — 알림 목록\n"
        "/schedule remove <번호> — 알림 삭제\n"
        "/model list|<이름> — 모델 관리\n"
        "/reset — 대화 초기화"
    )


async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    chat_id = str(update.effective_chat.id)

    if not args:
        await update.message.reply_text("사용법: /schedule add|interval|once|ai|list|remove")
        return

    action = args[0]

    if action == "list":
        schedules = load_schedules()
        mine = [s for s in schedules if s["chat_id"] == chat_id]
        if not mine:
            await update.message.reply_text("등록된 알림 없음")
            return
        lines = []
        for i, s in enumerate(mine):
            if s["type"] == "cron":
                time_str = f"매일 {s['hour']:02d}:{s['minute']:02d}"
            elif s["type"] == "interval":
                time_str = f"매 {s['minutes']}분"
            elif s["type"] == "once":
                time_str = f"일회 {s['run_at']}"
            ai_tag = " [AI]" if s.get("use_ai") else ""
            lines.append(f"{i+1}. {time_str}{ai_tag} — {s['message']}")
        await update.message.reply_text("\n".join(lines))
        return

    if action == "remove":
        if len(args) < 2:
            await update.message.reply_text("사용법: /schedule remove <번호>")
            return
        idx = int(args[1]) - 1
        schedules = load_schedules()
        mine = [s for s in schedules if s["chat_id"] == chat_id]
        if idx < 0 or idx >= len(mine):
            await update.message.reply_text("잘못된 번호")
            return
        target = mine[idx]
        try:
            scheduler.remove_job(target["id"])
        except Exception:
            pass
        schedules = [s for s in schedules if s["id"] != target["id"]]
        save_schedules(schedules)
        await update.message.reply_text(f"삭제: {target['message']}")
        return

    if action in ("add", "ai"):
        if len(args) < 3:
            await update.message.reply_text(f"사용법: /schedule {action} HH:MM <메시지>")
            return
        try:
            hour, minute = map(int, args[1].split(":"))
        except ValueError:
            await update.message.reply_text("시간 형식: HH:MM (예: 09:00)")
            return
        message = " ".join(args[2:])
        use_ai = action == "ai"
        job_id = f"cron_{chat_id}_{hour}_{minute}_{len(load_schedules())}"

        schedule_data = {
            "id": job_id,
            "type": "cron",
            "chat_id": chat_id,
            "hour": hour,
            "minute": minute,
            "message": message,
            "use_ai": use_ai,
        }

        scheduler.add_job(
            send_scheduled_message,
            CronTrigger(hour=hour, minute=minute, timezone="Asia/Seoul"),
            args=[chat_id, message, use_ai],
            id=job_id,
        )

        schedules = load_schedules()
        schedules.append(schedule_data)
        save_schedules(schedules)

        ai_tag = " (AI 응답)" if use_ai else ""
        await update.message.reply_text(f"매일 {hour:02d}:{minute:02d} 알림 등록{ai_tag}: {message}")
        return

    if action == "interval":
        if len(args) < 3:
            await update.message.reply_text("사용법: /schedule interval <분> <메시지>")
            return
        minutes = int(args[1])
        message = " ".join(args[2:])
        job_id = f"interval_{chat_id}_{minutes}_{len(load_schedules())}"

        schedule_data = {
            "id": job_id,
            "type": "interval",
            "chat_id": chat_id,
            "minutes": minutes,
            "message": message,
            "use_ai": False,
        }

        scheduler.add_job(
            send_scheduled_message,
            IntervalTrigger(minutes=minutes),
            args=[chat_id, message, False],
            id=job_id,
        )

        schedules = load_schedules()
        schedules.append(schedule_data)
        save_schedules(schedules)

        await update.message.reply_text(f"매 {minutes}분 알림 등록: {message}")
        return

    if action == "once":
        if len(args) < 3:
            await update.message.reply_text("사용법: /schedule once HH:MM <메시지>")
            return
        try:
            hour, minute = map(int, args[1].split(":"))
        except ValueError:
            await update.message.reply_text("시간 형식: HH:MM")
            return
        message = " ".join(args[2:])
        run_at = datetime.now().replace(hour=hour, minute=minute, second=0)
        if run_at < datetime.now():
            run_at += timedelta(days=1)
        job_id = f"once_{chat_id}_{len(load_schedules())}"

        schedule_data = {
            "id": job_id,
            "type": "once",
            "chat_id": chat_id,
            "run_at": run_at.isoformat(),
            "message": message,
            "use_ai": False,
        }

        scheduler.add_job(
            send_scheduled_message,
            DateTrigger(run_date=run_at),
            args=[chat_id, message, False],
            id=job_id,
        )

        schedules = load_schedules()
        schedules.append(schedule_data)
        save_schedules(schedules)

        await update.message.reply_text(f"일회 알림 등록: {run_at.strftime('%m/%d %H:%M')} — {message}")
        return

    await update.message.reply_text("사용법: /schedule add|interval|once|ai|list|remove")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or args[0] == "list":
        lines = []
        for name in manager.models:
            marker = " (현재)" if name == manager.current else ""
            lines.append(f"- {name}{marker}")
        await update.message.reply_text("\n".join(lines))
    else:
        result = manager.switch(args[0])
        await update.message.reply_text(result)


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
        {"messages": [HumanMessage(content=user_message)]}, cfg
    )

    suffix = f"\n[{used_model}]" if manager.auto_mode else ""
    await update.message.reply_text(response_text + suffix)


async def main():
    global bot_app

    BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    bot_app = Application.builder().token(BOT_TOKEN).build()

    bot_app.add_handler(CommandHandler("start", start))
    bot_app.add_handler(CommandHandler("schedule", schedule_command))
    bot_app.add_handler(CommandHandler("model", model_command))
    bot_app.add_handler(CommandHandler("reset", reset_command))
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 스케줄러 시작 + 저장된 스케줄 복원
    restore_schedules()
    scheduler.start()

    print(f"봇 시작 (스케줄러: {len(load_schedules())}개 복원)")
    await bot_app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
