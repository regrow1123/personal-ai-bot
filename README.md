# Personal AI Bot

텔레그램 개인 AI 봇 + 멀티 프로바이더 동적 관리

LangChain/LangGraph 학습 프로젝트

## 학습 단계

### 기초
- [ ] 1단계: 환경 세팅
- [ ] 2단계: 단일 LLM 챗봇
- [ ] 3단계: LangGraph 에이전트 + 메모리
- [ ] 4단계: 커스텀 도구 추가

### 텔레그램 봇
- [ ] 5단계: 텔레그램 봇 연동
- [ ] 6단계: ModelManager 구현
- [ ] 7단계: 동적 전환 + 자동 로테이션

### 에이전트 설정
- [ ] 8단계: 마크다운 기반 에이전트 설정
  - SOUL.md (봇 페르소나/성격)
  - USER.md (사용자 정보/선호도)
  - TOOLS.md (도구 설명)
  - MEMORY.md (장기 메모리)
  - 시스템 프롬프트에 자동 로드
  - /soul, /user 명령으로 텔레그램에서 편집

### 지식 + 기억
- [ ] 9단계: RAG — ChromaDB 기반 개인 지식베이스
- [ ] 10단계: 장기 메모리 — SQLite 대화 이력 + 요약

### 확장
- [ ] 11단계: MCP 서버 연동
- [ ] 12단계: 멀티모달 입력 — 이미지 분석, 음성 메시지 STT
- [ ] 13단계: 대화 비교 — /compare로 여러 모델 동시 응답 비교

### 멀티 에이전트
- [ ] 14단계: 역할별 에이전트 분리 + LangGraph 협업 그래프

### 운영
- [ ] 15단계: 스케줄러 + 알림 (APScheduler)
- [ ] 16단계: 사용량 대시보드 — 토큰/비용 추적
- [ ] 17단계: 웹 대시보드 — FastAPI + 웹 UI (선택)

## 기술 스택

- Python 3.x
- LangChain / LangGraph
- python-telegram-bot
- ChromaDB
- SQLite
- Ollama (로컬 LLM)
- Google Gemini API
- Anthropic Claude API (선택)

## 설치

```bash
pip install -r requirements.txt
```
