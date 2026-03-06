"""
4단계: 커스텀 도구 추가
- @tool 데코레이터로 도구 정의
- 에이전트가 필요에 따라 도구를 자동 호출 (ReAct 루프)
"""
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


# === 커스텀 도구 정의 ===

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
    # 간단한 예시 데이터 — 나중에 DB/파일로 교체
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

# 도구 목록을 에이전트에 전달
tools = [get_current_time, calculate, search_memo]
agent = create_react_agent(llm, tools=tools, checkpointer=memory)

config = {"configurable": {"thread_id": "my-chat-1"}}

print("도구 장착 에이전트 (종료: quit)")
print(f"사용 가능한 도구: {[t.name for t in tools]}")
print("-" * 40)

while True:
    user_input = input("\n나: ")
    if user_input.lower() in ("quit", "exit", "q"):
        print("종료")
        break

    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config
    )

    ai_message = response["messages"][-1]
    print(f"\nAI: {ai_message.content}")
