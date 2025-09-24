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

