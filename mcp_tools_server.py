"""
MCP 도구 서버
- 시간, 계산, 지식베이스 도구를 MCP 프로토콜로 제공
- 봇과 독립적으로 실행
"""
import json
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

KNOWLEDGE_FILE = Path(__file__).parent / "knowledge.json"

mcp = FastMCP("personal-tools")


@mcp.tool()
def get_current_time() -> str:
    """현재 날짜와 시간을 반환한다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def calculate(expression: str) -> str:
    """수학 계산식을 받아서 결과를 반환한다. 예: '2 + 3 * 4'"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@mcp.tool()
def save_note(title: str, content: str) -> str:
    """메모를 저장한다."""
    notes = {}
    if KNOWLEDGE_FILE.exists():
        notes = json.loads(KNOWLEDGE_FILE.read_text())

    notes[title] = {
        "content": content,
        "saved_at": datetime.now().isoformat()
    }
    KNOWLEDGE_FILE.write_text(json.dumps(notes, ensure_ascii=False, indent=2))
    return f"메모 저장 완료: {title}"


@mcp.tool()
def search_notes(keyword: str) -> str:
    """키워드로 메모를 검색한다."""
    if not KNOWLEDGE_FILE.exists():
        return "저장된 메모 없음"

    notes = json.loads(KNOWLEDGE_FILE.read_text())
    results = []
    for title, data in notes.items():
        if keyword in title or keyword in data["content"]:
            results.append(f"- {title}: {data['content']}")

    if results:
        return "\n".join(results)
    return f"'{keyword}' 관련 메모 없음"


if __name__ == "__main__":
    mcp.run()
