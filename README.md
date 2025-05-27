# DOE (Design of Experiments) for AI Prompt Engineering

## 프로젝트 개요

이 프로젝트는 **실험계획법(Design of Experiments)**을 활용하여 AI 프롬프트 엔지니어링의 효과를 체계적으로 분석하는 연구 도구입니다. 다양한 프롬프트 요인들이 AI 응답의 일관성과 품질에 미치는 영향을 통계적으로 검증합니다.

### 연구 목표
- AI 프롬프트 요인들의 **주효과(Main Effects)** 분석
- 요인 간 **교호작용(Interaction Effects)** 발견
- **비용 효율적인** 실험 설계를 통한 최적 프롬프트 전략 도출
- **재현 가능한** 실험 프레임워크 제공

## 프로젝트 구조

```
DOE/
├── 2kfactorial/           # 2^k Factorial Design 실험
│   ├── experiment_design.py
│   ├── prompt_generator.py
│   ├── response_collector.py
│   ├── consistency_analyzer.py
│   ├── factorial_analyzer.py
│   ├── main_experiment.py
│   ├── run_experiment.py
│   ├── power_analysis.py
│   ├── config.py
│   ├── requirements.txt
│   └── README.md
│
├── rcbd/                  # Randomized Complete Block Design 실험
│   ├── experiment_design.py
│   ├── prompt_generator.py
│   ├── response_collector.py
│   ├── consistency_analyzer.py
│   ├── rcbd_analyzer.py
│   ├── main_experiment.py
│   ├── run_experiment.py
│   ├── config.py
│   ├── requirements.txt
│   └── README.md
│
├── factorial_results/     # 2^k 실험 결과
├── rcbd_results/         # RCBD 실험 결과
├── requirements.txt      # 전체 프로젝트 의존성
├── .gitignore           # Git 제외 파일 설정
└── README.md           # 이 파일
```

## 실험 설계 방법론

### 1. 2^k Factorial Design (완전요인설계)
**위치**: `2kfactorial/`

**특징**:
- 모든 요인의 모든 조합을 테스트
- 주효과와 교호작용을 동시에 분석
- 통계적으로 강력한 결론 도출

**실험 요인** (2^4 = 16 조합):
1. **프롬프트 언어** (korean/english)
2. **AI 모델** (gpt-3.5-turbo/gpt-4o-mini)
3. **역할 부여** (with_role/no_role)
4. **명시성** (high/low)

**최적화 결과**:
- ANOVA 분석을 통해 `context_provision` 요인 제거 (p=0.982)
- **50% 비용 절약** 달성 (2^5 → 2^4)

### 2. Randomized Complete Block Design (RCBD)
**위치**: `rcbd/`

**특징**:
- 블록 요인(질문 카테고리)의 영향을 통제
- 처리 요인의 순수한 효과 측정
- 실험 오차 감소

**블록 요인**: 질문 카테고리 (인성, 창의성, 논리적추론)
**처리 요인**: 프롬프트 구성 요소들

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd DOE

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는 venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 2. 실험 실행

#### 2^k Factorial Design 실험
```bash
cd 2kfactorial

# 데모 실험 (빠른 테스트)
python run_experiment.py --mode demo

# 전체 실험 (완전한 분석)
python run_experiment.py --mode full

# 실험 정보 확인
python run_experiment.py --info-only
```

#### RCBD 실험
```bash
cd rcbd

# 기본 실험 실행
python run_experiment.py

# 설정 확인
python run_experiment.py --info-only
```

### 3. Power Analysis (표본 크기 결정)
```bash
cd 2kfactorial

# Power analysis 실행
python power_analysis.py
```

## 실험 모드 비교

### 2^k Factorial Design 모드

| 모드 | 조건 수 | API 호출 | 예상 비용 | 소요 시간 | 용도 |
|------|---------|----------|-----------|-----------|------|
| **demo** | 12 | 60 | ~$0.03 | 2-3분 | 빠른 테스트 |
| **test** | 36 | 180 | ~$0.08 | 5-7분 | 요인 효과 확인 |
| **representative** | 48 | 240 | ~$0.32 | 8-10분 | 효율적 분석 |
| **full** | 240 | 1,200 | ~$2.24 | 30-40분 | 완전한 실험 |

### RCBD 모드

| 설정 | 블록 수 | 처리 수 | API 호출 | 예상 비용 | 특징 |
|------|---------|---------|----------|-----------|------|
| **기본** | 3 | 8 | 120 | ~$0.15 | 블록 효과 통제 |
| **확장** | 3 | 16 | 240 | ~$0.30 | 더 많은 처리 조합 |

## 분석 결과 및 출력

### 자동 생성 파일

#### 실험 데이터
- `*_experiment_data_*.csv`: 전체 실험 원시 데이터
- `*_analysis_data_*.csv`: 통계 분석용 정제 데이터
- `*_experiment_summary_*.json`: 실험 요약 통계

#### 분석 결과
- `*_factorial_analysis_*.html`: ANOVA 결과 및 시각화
- `*_power_analysis_*.html`: Power analysis 결과
- `*_consistency_analysis_*.json`: 응답 일관성 분석

#### 중간 결과 (선택적)
- `01_*_prompts_generated.json`: 생성된 프롬프트
- `02_*_responses_collected.json`: 수집된 AI 응답
- `03_*_consistency_analyzed.json`: 일관성 분석 결과

### 주요 분석 지표

1. **응답 일관성**: BERT Multilingual 기반 BERTScore
2. **주효과**: 각 요인의 독립적 영향
3. **교호작용**: 요인 간 상호작용 효과
4. **블록 효과**: 질문 카테고리의 영향 (RCBD)
5. **효과 크기**: Cohen's d, Eta-squared