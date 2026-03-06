"""
3단계: LangGraph 에이전트 + 메모리
- create_react_agent로 에이전트 구조 전환
- MemorySaver로 대화 히스토리 유지
- thread_id 기반 세션 관리
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# LLM
llm = ChatOllama(model="qwen3:4b")

# 메모리 (대화 이력 자동 관리)
memory = MemorySaver()

# 에이전트 생성 (도구는 아직 없음 — 4단계에서 추가)
agent = create_react_agent(llm, tools=[], checkpointer=memory)

# 세션 설정 — thread_id가 같으면 같은 대화로 이어짐
config = {"configurable": {"thread_id": "my-chat-1"}}

print("에이전트 챗봇 시작 (종료: quit)")
print("-" * 40)

while True:
    user_input = input("\n나: ")
    if user_input.lower() in ("quit", "exit", "q"):
        print("종료")
        break

    # 에이전트 호출 — messages append를 에이전트가 자동으로 해준다
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config
    )

    # 마지막 메시지가 AI 응답
    ai_message = response["messages"][-1]
    print(f"\nAI: {ai_message.content}")
