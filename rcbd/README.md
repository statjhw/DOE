# RCBD 감정 프레이밍 실험

## 📋 실험 개요

**실험 목표**: 감정 프레이밍에 따라 AI 응답의 일관성이 달라지는지 분석

**실험 규모**: 총 351개 데이터 포인트 (3 카테고리 × 39 질문 × 3 프레이밍)

### 🎯 실험 설계 (RCBD)

- **설계**: Randomized Complete Block Design (완전확률화블록설계)
- **독립변수 (처리요인)**: 프레이밍 수준 (3수준)
  - 중립적 표현: 감정 유도 없이 사실 전달 중심
  - 정서적 표현: 긍정/부정의 온건한 감정 포함
  - 자극적 표현: 강한 비판, 갈등, 위협 등 포함

- **블록 요인**: 질문 카테고리 (3개 블록, 각 39개 질문)
  - 인성 (윤리/정의 포함) - 추상적 도덕 원칙부터 구체적 실생활 딜레마까지
  - 창의성 - 아이디어 발상부터 혁신적 문제해결까지
  - 논리적 추론 - 논리적 사고부터 복잡한 의사결정까지

- **종속변수**: AI 응답 일관성
  - **측정 지표**: BERT Multilingual 기반 BERTScore
  - 모든 응답 쌍 간 유사도의 평균값 사용

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
# 데모 실험 (2-3분, $0.5-1)
python run_experiment.py --mode demo

# 테스트 실험 (5-10분, $1-3)
python run_experiment.py --mode test

# 전체 실험 (2-4시간, $0.5-1)
python run_experiment.py --mode full
```

## 📁 프로젝트 구조

```
rcbd/
├── experiment_design.py     # 실험 설계 정의
├── prompt_generator.py      # 프레이밍 프롬프트 생성
├── response_collector.py    # AI 응답 수집
├── consistency_analyzer.py  # 일관성 분석
├── rcbd_analyzer.py        # RCBD 통계 분석
├── main_experiment.py      # 메인 실험 실행
├── run_experiment.py       # 실행 스크립트
├── config.py              # 설정 관리
├── requirements.txt       # 패키지 의존성
└── README.md             # 이 파일
```

## 🔬 실험 프로세스

### 1단계: 프레이밍 프롬프트 생성
- 각 기본 질문에 대해 3가지 프레이밍 수준의 변형 생성
- GPT-4o-mini를 사용하여 자동 생성

### 2단계: AI 응답 수집
- 각 프레이밍된 프롬프트에 대해 여러 개의 응답 수집
- 동일한 프롬프트에 대한 일관성 측정을 위함

### 3단계: 일관성 분석
- **BERT Multilingual BERTScore**: Google의 bert-base-multilingual-cased 모델 기반 유사도
- 모든 응답 쌍에 대해 BERTScore 계산 후 평균값 사용
- 한국어를 포함한 다국어 의미적 유사도 측정

### 4단계: RCBD 통계 분석
- 이원 분산분석(Two-way ANOVA)
- 프레이밍 효과와 블록 효과 분석
- 교호작용 효과 검정

## 📊 결과 해석

### 주요 가설
- **H1**: 프레이밍 수준에 따라 AI 응답의 일관성이 다를 것이다
- **H2**: 질문 카테고리(블록)에 따라 기준 일관성이 다를 것이다

### 예상 결과
- 중립적 프레이밍 → 높은 일관성
- 자극적 프레이밍 → 낮은 일관성 (더 다양한 응답)
- 논리적추론 → 높은 일관성
- 창의성 → 낮은 일관성

## 📈 출력 파일

### 실험 데이터
- `rcbd_experiment_data_TIMESTAMP.csv`: 전체 실험 데이터
- `rcbd_analysis_data_TIMESTAMP.csv`: RCBD 분석용 데이터
- `experiment_summary_TIMESTAMP.json`: 요약 통계

### 중간 결과 (중간 저장 기능)
- `01_test_prompts_generated.json`: 생성된 프롬프트
- `02_test_responses_collected.json`: 수집된 응답
- `03_test_consistency_analyzed.json`: 일관성 분석 결과

## ⚙️ 설정 옵션

### config.py 주요 설정
```python
DEFAULT_EXPERIMENT_PARAMS = {
    'n_responses': 4,          # 각 프롬프트당 응답 수
    'temperature': 0.7,        # 응답 다양성
    'max_tokens': 500,         # 최대 토큰 수
    'api_delay': 1.0,          # API 호출 간격 (초)
}

# 분석 설정
BERT_MULTILINGUAL_MODEL = 'bert-base-multilingual-cased'
BERTSCORE_LANG = 'ko'
```

### 실험 모드
- **demo**: 최소 데이터로 빠른 테스트
- **test**: 각 카테고리당 1개 질문으로 테스트
- **full**: 전체 데이터로 완전한 실험

## 🛠️ 개발자 가이드

### 새로운 프레이밍 수준 추가

1. `experiment_design.py`에서 `FRAMING_DEFINITIONS` 수정
2. `prompt_generator.py`에서 프레이밍 로직 업데이트

### 새로운 측정 지표 추가

1. `consistency_analyzer.py`에 새 메서드 추가
2. `rcbd_analyzer.py`에서 분석 로직 업데이트

### 커스텀 질문 세트 사용

```python
# experiment_design.py 수정
CUSTOM_QUESTION_BLOCKS = {
    "새카테고리": [
        "새로운 질문 1",
        "새로운 질문 2",
        "새로운 질문 3"
    ]
}
```

## 📚 이론적 배경

### RCBD (Randomized Complete Block Design)
- 블록 요인으로 이질성 제거
- 처리 효과의 정확한 추정
- 실험 효율성 증대

### 일관성 측정
- **BERT Multilingual BERTScore**: Google의 bert-base-multilingual-cased 모델 기반 유사도 측정
- 사전 훈련된 BERT Multilingual 모델로 한국어 텍스트의 의미적 유사도 계산
- 다국어 지원으로 한국어 텍스트에 대한 안정적인 성능 제공

## 🤝 기여하기

1. 이슈 생성 또는 기존 이슈 확인
2. 브랜치 생성: `git checkout -b feature/새기능`
3. 변경사항 커밋: `git commit -m '새기능 추가'`
4. 브랜치 푸시: `git push origin feature/새기능`
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 🔗 참고 자료

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [BERT Multilingual Model](https://huggingface.co/bert-base-multilingual-cased)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [RCBD in Design of Experiments](https://en.wikipedia.org/wiki/Randomized_complete_block_design) 