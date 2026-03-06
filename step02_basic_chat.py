"""
2단계: 단일 LLM 챗봇
- Ollama 로컬 모델로 기본 대화
- SystemMessage, HumanMessage 사용법
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Ollama 로컬 모델 연결
llm = ChatOllama(model="llama3.2")

# 대화 이력 관리
messages = [
    SystemMessage(content="너는 친절한 개인 AI 어시스턴트다. 한국어로 답변한다.")
]

print("챗봇 시작 (종료: quit)")
print("-" * 40)

while True:
    user_input = input("\n나: ")
    if user_input.lower() in ("quit", "exit", "q"):
        print("종료")
        break

    # 사용자 메시지 추가
    messages.append(HumanMessage(content=user_input))

    # LLM 호출
    response = llm.invoke(messages)

    # 응답을 이력에 추가
    messages.append(response)

    print(f"\nAI: {response.content}")
