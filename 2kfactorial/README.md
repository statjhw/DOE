# 2k Factorial Design AI 프롬프트 실험 (최적화 버전)

## 📋 실험 개요

**실험 목표**: 4개 요인이 AI 응답 일관성에 미치는 주효과와 교호작용 분석

**실험 설계**: 2^4 Factorial Design (context_provision 제거)

### 🎯 실험 요인 (2^4 = 16 조합)

1. **프롬프트 언어** (korean/english) - 응답은 항상 한국어
2. **AI 모델** (gpt-3.5-turbo/gpt-4o-mini)
3. **역할 부여** (with_role/no_role) 
4. **명시성** (high/low)

### ❌ 제거된 요인
- **맥락 제공**: ANOVA 결과 p=0.982로 효과 없음 (50% 비용 절약)

### 🧱 블록 요인
- **질문 카테고리**: 인성, 창의성, 논리적추론 (각 5개 질문)

### 📊 종속변수
- **응답 일관성**: BERT Multilingual 기반 BERTScore 유사도

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정 (.env 파일 생성)
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 2. 실험 실행

```bash
# 데모 실험 (2개 요인, 12조건, ~$0.01)
python run_experiment.py --mode demo

# 테스트 실험 (3개 요인, 36조건, ~$0.05)
python run_experiment.py --mode test

# 대표 설계 (4개 요인, 48조건, ~$0.20)
python run_experiment.py --mode representative

# 전체 설계 (4개 요인, 240조건, ~$1.00)
python run_experiment.py --mode full
```

### 3. 실험 정보 확인

```bash
# 모든 모드 정보 출력
python run_experiment.py --info-only
```

## 📁 프로젝트 구조

```
2kfactorial/
├── experiment_design.py     # 2^4 요인 설계 정의
├── prompt_generator.py      # 요인 조합별 프롬프트 생성
├── response_collector.py    # AI 응답 수집
├── consistency_analyzer.py  # BERT Multilingual 일관성 분석
├── factorial_analyzer.py    # 2k Factorial 통계 분석
├── main_experiment.py      # 메인 실험 실행기
├── run_experiment.py       # 실행 스크립트
├── config.py              # 설정 관리
├── requirements.txt       # 패키지 의존성
└── README.md             # 이 파일
```

## 🔬 실험 설계 상세

### Representative vs Full Design

| 설계 방식 | 조건 수 | 특징 |
|----------|---------|------|
| **Representative** | 48개 | 16개 요인조합 × 3개 카테고리<br/>카테고리당 1개 대표질문 |
| **Full** | 240개 | 16개 요인조합 × 15개 전체질문<br/>카테고리를 블록으로 처리 |

### 요인별 수준 설명

#### 1. 프롬프트 언어 (prompt_language)
- **korean**: 한국어 프롬프트 + "한국어로 답변해 주세요."
- **english**: 영어 프롬프트 + "Please respond in Korean."

#### 2. AI 모델 (model)
- **gpt-3.5-turbo**: GPT-3.5 Turbo
- **gpt-4o-mini**: GPT-4o Mini

#### 3. 역할 부여 (role_assignment)
- **with_role**: "당신은 [카테고리별 전문가]입니다. [질문]"
- **no_role**: "[질문]"

#### 4. 명시성 (explicitness)
- **high**: "[질문] 구체적이고 명확한 예시와 함께 자세히 설명해 주세요."
- **low**: "[질문]"

## 📊 분석 방법

### 1. 주효과 분석
각 요인이 응답 일관성에 미치는 독립적 영향 분석

### 2. 교호작용 분석
요인 간 상호작용 효과 분석 (2차 교호작용)

### 3. 블록 효과 분석
질문 카테고리가 일관성에 미치는 영향 분석

### 4. ANOVA 최적화 결과
- **유의한 요인**: model (p<0.001), explicitness (p<0.001), category (p<0.05)
- **제거된 요인**: context_provision (p=0.982, 효과 없음)

## 💰 비용 추정 (50% 절약!)

| 모드 | 조건 수 | API 호출 | 예상 비용 | 소요 시간 |
|------|---------|----------|-----------|-----------|
| **demo** | 12 | 36 | ~$0.01 | 2-3분 |
| **test** | 36 | 108 | ~$0.05 | 5-7분 |
| **representative** | 48 | 144 | ~$0.20 | 8-10분 |
| **full** | 240 | 720 | ~$1.00 | 30-40분 |

## 📈 출력 파일

### 실험 데이터
- `factorial_experiment_data_{mode}_{timestamp}.csv`: 전체 실험 데이터
- `factorial_analysis_data_{mode}_{timestamp}.csv`: 분석용 데이터
- `factorial_experiment_summary_{mode}_{timestamp}.json`: 요약 통계

### 중간 결과 (save_intermediate=True)
- `01_{mode}_prompts_generated.json`: 생성된 프롬프트
- `02_{mode}_responses_collected.json`: 수집된 응답
- `03_{mode}_consistency_analyzed.json`: 일관성 분석 결과

## ⚙️ 설정 옵션

### 실험 매개변수 (config.py)
```python
DEFAULT_EXPERIMENT_PARAMS = {
    'n_responses': 3,           # 각 조건당 응답 수
    'temperature': 0.7,         # 응답 다양성
    'max_tokens': 500,          # 최대 토큰 수
    'api_delay': 1.0,           # API 호출 간격 (초)
}
```

### 모델 설정
```python
BERT_MULTILINGUAL_MODEL = 'bert-base-multilingual-cased'
BERTSCORE_LANG = 'ko'
```

## 🧪 실험 예시

### 프롬프트 변형 예시
**기본 질문**: "어려움에 처한 친구를 도와야 하는 이유는 무엇인가?"

**요인 조합에 따른 변형**:
- **Korean + Direct + No Role + No Context + Low**: 
  "어려움에 처한 친구를 도와야 하는 이유는 무엇인가? 한국어로 답변해 주세요."

- **English + Indirect + With Role + With Context + High**:
  "You are a professional counselor. Many people in modern society are struggling with this issue. What are your thoughts on this: What are the reasons for helping a friend in difficulty? Please provide a detailed explanation with specific and clear examples. Please respond in Korean."

## 📚 이론적 배경

### 2k Factorial Design
- **완전요인설계**: 모든 요인 조합을 포함
- **주효과**: 각 요인의 독립적 영향
- **교호작용**: 요인 간 상호작용 효과
- **효율성**: 적은 실험으로 많은 정보 획득

### BERT Multilingual 일관성 측정
- **BERTScore**: 사전훈련된 BERT 모델 기반 의미적 유사도
- **다국어 지원**: 한국어 텍스트에 최적화
- **쌍별 비교**: 모든 응답 쌍의 유사도 평균

## 🔧 커스터마이징

### 새로운 요인 추가
1. `experiment_design.py`의 `FACTORS` 딕셔너리에 요인 추가
2. `FACTOR_TEMPLATES`에 프롬프트 템플릿 정의
3. `config.py`의 `FACTOR_INFO`에 요인 정보 추가

### 새로운 질문 추가
```python
# experiment_design.py
QUESTION_BLOCKS = {
    '새카테고리': [
        "새로운 질문 1",
        "새로운 질문 2"
    ]
}
```

### 새로운 분석 지표 추가
1. `consistency_analyzer.py`에 새 메서드 추가
2. `factorial_analyzer.py`에서 분석 로직 업데이트

## 🤝 기여하기

1. 이슈 생성 또는 기존 이슈 확인
2. 브랜치 생성: `git checkout -b feature/새기능`
3. 변경사항 커밋: `git commit -m '새기능 추가'`
4. 브랜치 푸시: `git push origin feature/새기능`
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🔗 참고 자료

- [2k Factorial Design Theory](https://en.wikipedia.org/wiki/Factorial_experiment)
- [BERT Multilingual Model](https://huggingface.co/bert-base-multilingual-cased)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [OpenAI API Documentation](https://platform.openai.com/docs) 